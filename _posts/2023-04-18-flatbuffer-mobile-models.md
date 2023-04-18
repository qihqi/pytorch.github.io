Blazingly Fast Pytorch Mobile Model Loading with Flatbuffer
TL; DR;
With the aim of reducing the overall model loading time experienced in mobile applications, we added another serialization format based on flatbuffers.
We observed that a setup using flatbuffer in a single file can be loaded 10x+ faster compared to the pickle based format, a step function needed to meet mobile use cases.
Why?
Time to first inference is critical in many edge applications, which aim at more interactive user experiences, and thus track the critical metric of “user perceived latency”.
Interactions like a voice action (e.g. “take a picture”) would be degraded meaningfully by a delayed response. Thus, it’s common to set a high bar (e.g. 200ms) for the maximum permissible response time. This becomes even more important in wearable devices, where due to resource constraints it is common for applications to be loaded / unloaded regularly.
What do we cover in this blog?
At a high level, we have splitted mode loading into 3 steps: Read from storage, Deserialization, and Runtime initialization. This potentially allow amortizing the cost of them across different times in an application lifecycle; plus provide a nice separation of concerns between model representation/deserialization and runtime initialization.
The read from storage part simply means reading the file from disk to memory as raw bytes. Deserialization means creating manipulable in-memory structures from the raw bytes. And finally runtime initialization means creating the in-memory torch::jit::mobile::Module ready for inference.
Benchmarks
In our benchmarks, we load a set of models in different mobile devices, using both pickle and flatbuffer format, and measure the latency.
The models are loaded in C++ using the torch::jit::mobile::_load_for_mobile(path) API.
Models used for test:
Resnet50 as available with model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True). This model has a file size of 161MB
Resnet101 as available with model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True); this model has a file size of 234MB
And Silero speech-to-text as available at model, _,_ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language='en')this model has a file size of 112MB
Devices used to test are iPhone X, iPhone 13 Pro, Pixel 4 and Pixel 6 Pro.
Results:
Resnet 50:

device
flatbuffer mean (ms)
flatbuffer p90 (ms)
pickle mean (ms)
pickle p90
mean ratio
p90 ratio
iPhone X
4.23
4.26
74.24
75.05
17.55
17.62
iPhone 13 Pro
2.60
2.64
44.72
44.80
17.19
17.00
Pixel 4
4.23
4.44
99.02
101.13
23.39
22.78
Pixel 6 Pro
3.13
3.18
77.92
79.98
24.86
25.14





Resnet 101:

device
flatbuffer mean(ms)
flatbuffer p90 (ms)
pickle mean(ms)
pickle p90(ms)
mean ratio
p90 ratio
iPhone X
7.31
7.45
106.72
107.53
14.58
14.43
iPhone 13 Pro
4.17
4.22
64.60
64.81
15.49
15.36
Pixel 4
6.07
6.21
142.35
146.51
23.43
23.59
Pixel 6 Pro
4.42
4.48
101.42
104.41
22.97
23.33



Sileto TTS

device
flatbuffer mean(ms)
flatbuffer p90(ms)
pickle mean(ms)
pickle p90(ms)
mean ratio
p90 ratio
iPhone X
5.37
5.46
56.79
57.14
10.57
10.47
iPhone 13 Pro
3.16
3.20
33.54
33.59
10.62
10.48
Pixel 4
4.35
4.47
78.48
79.86
18.03
17.87
Pixel 6 Pro
3.39
3.43
57.02
58.69
16.80
17.11



Why is it so fast?

Deserialize for flatbuffer is almost free.
Recall the 3 stages of model loading: read from disk, deserialization and initialization. The read from disk stage for both flatbuffer and pickle format is the same, because the file sizes of those two formats are very similar. However, deserialization for pickle means running the Unpickler, which is a stack machine interpreting pickle bytecodes. After unpickling, A tuple of torch::jit::IValues are created.

Flatbuffer files has this property that the on-disk layout matches its in-memory layout. Therefore, the deserialize part, which is provided by the flatbuffer library, is just pointer arithmetic followed by a cast:
(https://github.com/google/flatbuffers/blob/3fda20d7c7fe1f8006210bddae8cb55bc7a74c3b/include/flatbuffers/buffer.h#L132) Of course, accessing individual fields inside of casted Flatbuffer object also has some CPU computations, such as additional pointer arithmetic and endianness checks, but such operation are much faster than running pickler’s stack machine.


template<typename T> T *GetMutableRoot(void *buf) {
  EndianCheck();
  return reinterpret_cast<T *>(
      reinterpret_cast<uint8_t *>(buf) +
      EndianScalar(*reinterpret_cast<uoffset_t *>(buf)));
}



Flatbuffer’s initialize part is also more efficient:

A Pytorch model consists of two parts:
Tensor weights of the model
Model’s code serialized as stack machine bytecode that the torchscript’s lite interpreter executes when the model runs on device. Along with non-tensor constants (such as string class names, precomputed error messages etc).

Loading of the weight itself is just reading bytes from disk to memory and should be the same for both pickle and flatbuffer formats.

However, when loading instructions, flatbuffer format has major advantages.

In flatbuffer format, instruction is stored in the Instruction flatbuffer struct, whose generated code looks like the following:
FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(4) Instruction FLATBUFFERS_FINAL_CLASS {
private:
 int8_t op_;
 int8_t padding0__;
 uint16_t n_;
 int32_t x_;
.......
}
Which happens to be the same layout of struct Instruction  defined in Pytorch. So parsing this instruction is a memcpy away.
With pickle, however, every instruction is serialized as a tuple of (string opcode, int n, int x); This tuple is then pickled. So to deserialize, we first unpickle to get this tuple back, then construct an Instruction struct by converting string OpCode -> int8 opcode.

How to create a flatbuffer model file:
When we save a mobile::Module we can pass the optional _use_flatbuffer argument and it will produce flatbuffer format:

Example (python):

>>> m = torch.jit.load(...)  # m is a ScriptModule
>>> m._save_for_lite_interpreter('/tmp/hello.ff', _use_flatbuffer=True)

Or, in C++:

torch::jit::Module m = ...
bool _use_flatbuffer = true;
m._save_for_mobile(
    filename, _extra_files, _save_mobile_debug_info, _use_flatbuffer);

The loading side is unchanged:
_load_for_lite_interpreter (python) or _load_for_mobile (C++, and its corresponding bindings) will just work regardless which format it loads.
Conclusion:
We present a new file format that significantly speeds up loading of a model.

