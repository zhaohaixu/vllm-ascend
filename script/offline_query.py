# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

from vllm import LLM, SamplingParams

# os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
# os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "0"
# os.environ["VLLM_ENC_DYNAMIC_STREAM"] = "1"
os.environ["VLLM_ENC_ENABLE"] = "chacha20-naive"
os.environ["TORCH_ENC_ENABLE"] = "chacha20-naive"

# Sample prompts.
prompts = [
    "Hello, my name is Liam, a software engineer specializing in AI systems and distributed computing with over five years of experience in optimizing large language model (LLM) deployment. I grew up in a tech-focused family where my father, a former computer science professor, introduced me to programming at the age of 12, sparking a lifelong passion for building efficient, secure, and scalable technical solutions. After earning a master’s degree in Computer Engineering from Stanford University, I joined a leading AI startup where I’ve focused on bridging the gap between cutting-edge LLM research and real-world industrial applications—with a particular focus on vLLM, the high-throughput inference engine that has revolutionized how we serve large models at scale.​ My work primarily revolves around optimizing LLM deployment in distributed environments, addressing core challenges like performance bottlenecks, security risks, and resource efficiency. One of my most impactful projects involved integrating confidential computing into vLLM’s service 化 architecture, aiming to resolve the tension between data privacy and inference speed that plagues cloud-based LLM services. As I delved into the project, I spent months analyzing how GPU enclaves (such as those on NVIDIA H100) interact with vLLM’s memory management system, discovering that the dual burden of memory swapping and data encryption was crippling throughput for models larger than 30B parameters. To mitigate this, my team and I developed a set of profiling tools tailored to vLLM’s service 化 workflow—including custom timeline JSON exporters that track GPU kernel execution, CPU-GPU data transfer, and pickle serialization/deserialization latency—allowing us to identify and optimize key bottlenecks in the broadcast_object_list function and other critical distributed communication components.​ Beyond technical optimization, I’m deeply passionate about making LLM technology accessible and secure for all users. I’ve contributed to open-source projects like vLLM’s profiler module, writing documentation and sample scripts to help other engineers implement performance monitoring in their service 化 deployments. I also frequently speak at industry conferences, where I share insights on balancing confidentiality and performance in LLM cloud services—drawing from hands-on experience with tools like NVIDIA Nsight Systems, Prometheus, and Chrome DevTools for timeline visualization. In my free time, I enjoy hiking in the Sierra Nevada mountains, experimenting with homebrewed machine learning models on my personal GPU cluster, and mentoring computer science students from underrepresented backgrounds, helping them explore careers in AI systems engineering.​ What drives me most is the opportunity to build technology that empowers people while protecting their privacy. In an era where LLMs are increasingly integrated into healthcare, finance, and education, ensuring these systems are both efficient and secure is not just a technical challenge—it’s a ethical imperative. Whether I’m debugging a vLLM profiler timeline, optimizing a distributed broadcast protocol, or collaborating with security researchers to harden GPU enclave implementations, I bring a meticulous, user-centric approach to every project. I believe that the best technical solutions emerge from a deep understanding of both the underlying technology and the real-world problems it aims to solve—and I’m constantly striving to expand my knowledge in areas like cryptography, distributed systems, and LLM inference optimization to better meet those challenges.​ Looking ahead, I’m excited to explore the intersection of LLM service 化 and edge computing, where the constraints of limited hardware resources and strict latency requirements present new opportunities for innovation. I’m also eager to continue advocating for safer, more transparent AI deployment practices, working with industry leaders to establish standards for confidential LLM computing that balance performance, security, and usability. Whether through writing code, sharing knowledge, or mentoring the next generation of engineers, I’m committed to advancing the field of AI systems in a way that benefits society as a whole—one optimized deployment, one secure broadcast, and one well-profiled timeline at a time.",
]

sampling_params = SamplingParams(max_tokens=128, temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(model="/mnt/Qwen3-8B-W8A8",
              tensor_parallel_size=2,
              pipeline_parallel_size=1,
              trust_remote_code=True,
              max_model_len=32768,
              max_num_seqs=8,
              quantization="ascend",
              compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
            #   enforce_eager=True
              )

    print("start profiler................")
    # llm.start_profile()

    print("start generation................")
    outputs = llm.generate(prompts, sampling_params)
    print("end generation................")

    # llm.stop_profile()
    print("end profiler................")

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()