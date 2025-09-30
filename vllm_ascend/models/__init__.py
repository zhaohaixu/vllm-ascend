from vllm import ModelRegistry

import vllm_ascend.envs as envs


def register_model():
    from .deepseek_dbo import CustomDeepseekDBOForCausalLM  # noqa: F401
    from .deepseek_mtp import CustomDeepSeekMTP  # noqa: F401
    from .deepseek_v2 import CustomDeepseekV2ForCausalLM  # noqa: F401
    from .deepseek_v2 import CustomDeepseekV3ForCausalLM  # noqa: F401
    from .open_pangu import PanguUltraMoEForCausalLM  # noqa: F401
    from .open_pangu import PanguEmbeddedForCausalLM  # noqa: F401
    from .qwen2_5_vl import \
        AscendQwen2_5_VLForConditionalGeneration  # noqa: F401
    from .qwen2_vl import AscendQwen2VLForConditionalGeneration  # noqa: F401

    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "vllm_ascend.models.deepseek_mtp:CustomDeepSeekMTP")

    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_ascend.models.qwen2_vl:AscendQwen2VLForConditionalGeneration")

    if envs.USE_OPTIMIZED_MODEL:
        ModelRegistry.register_model(
            "Qwen2_5_VLForConditionalGeneration",
            "vllm_ascend.models.qwen2_5_vl:AscendQwen2_5_VLForConditionalGeneration"
        )
    else:
        ModelRegistry.register_model(
            "Qwen2_5_VLForConditionalGeneration",
            "vllm_ascend.models.qwen2_5_vl_without_padding:AscendQwen2_5_VLForConditionalGeneration_Without_Padding"
        )

    if envs.VLLM_ASCEND_ENABLE_DBO:
        ModelRegistry.register_model(
            "DeepseekV2ForCausalLM",
            "vllm_ascend.models.deepseek_dbo:CustomDeepseekDBOForCausalLM")

        ModelRegistry.register_model(
            "DeepseekV3ForCausalLM",
            "vllm_ascend.models.deepseek_dbo:CustomDeepseekDBOForCausalLM")

    else:
        ModelRegistry.register_model(
            "DeepseekV2ForCausalLM",
            "vllm_ascend.models.deepseek_v2:CustomDeepseekV2ForCausalLM")

        ModelRegistry.register_model(
            "DeepseekV3ForCausalLM",
            "vllm_ascend.models.deepseek_v2:CustomDeepseekV3ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "vllm_ascend.models.qwen3_moe:CustomQwen3MoeForCausalLM")

    ModelRegistry.register_model(
        "PanguProMoEForCausalLM",
        "vllm_ascend.models.pangu_moe:PanguProMoEForCausalLM")
    
    ModelRegistry.register_model(
        "PanguUltraMoEForCausalLM",
        "vllm_ascend.models.open_pangu:PanguUltraMoEForCausalLM")

    ModelRegistry.register_model(
        "PanguEmbeddedForCausalLM",
        "vllm_ascend.models.open_pangu:PanguEmbeddedForCausalLM")
