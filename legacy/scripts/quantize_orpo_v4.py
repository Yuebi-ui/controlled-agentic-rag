#!/usr/bin/env python3
"""
Phase 0 Step 2: AWQ INT4 量化合并后的 ORPO v4 BF16 模型

使用与现有 SFT INT4 相同的量化配置：
  bits=4, group_size=128, zero_point=true, quant_method=awq

基于 AutoAWQ 进行量化。
"""
import os, sys, time, argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM

PROJECT_DIR = "/root/autodl-tmp/rag-server"
MERGED_BF16_PATH = os.path.join(PROJECT_DIR, "LLaMA-Factory-main/output/qwen3_orpo_v4_bf16")
OUTPUT_INT4_PATH = os.path.join(PROJECT_DIR, "LLaMA-Factory-main/output/qwen3_orpo_v4_int4")

# 量化配置，与现有 SFT INT4 保持一致
QUANT_CONFIG = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "gemm",
}

# 校准样本：使用 Tesla 手册中的代表性文本
CALIBRATION_SAMPLES = [
    "Model 3 支持三种钥匙类型：认证手机、钥匙卡和遥控钥匙。您可以使用任意一种钥匙来解锁和驾驶车辆。",
    "要启用哨兵模式，请点击触摸屏顶部的哨兵模式图标，或者通过手机应用程序远程开启。",
    "自动辅助驾驶功能包括交通感知巡航控制和自动辅助转向，可以在高速公路和城市道路上使用。",
    "当电池电量降至 20% 以下时，车辆会自动提醒您充电，并在地图上显示附近的超级充电站。",
    "要调节方向盘位置，请使用方向盘左侧的控制杆。上下移动控制杆可以调节方向盘的高度。",
    "空调系统支持分区温度控制，驾驶员和前排乘客可以设置不同的温度。",
    "要使用语音命令，请按方向盘右侧的语音按钮，然后说出您的指令。",
    "行李箱可以通过触摸屏、手机应用程序或后备箱上的按钮打开。",
    "在寒冷天气中，建议在出发前使用手机应用程序预热车辆和电池。",
    "如果轮胎压力过低，仪表盘会显示黄色警告灯，并在触摸屏上显示具体的轮胎位置。",
    "要连接蓝牙设备，请在触摸屏上进入蓝牙设置，选择添加新设备，然后在手机上确认配对码。",
    "车辆在行驶过程中会自动锁定所有车门，您可以在设置中更改自动锁定的速度阈值。",
    "牵引力控制系统可以帮助防止车轮打滑，在湿滑路面上提高车辆的稳定性。",
    "要检查车辆的软件版本，请进入控制 → 软件，当前版本号会显示在屏幕上。",
    "儿童安全锁位于后车门的侧面，开启后车内无法从内部打开后车门。",
    "如果安全气囊系统检测到碰撞，车辆会自动切断高压电源并打开危险警告灯。",
    "要激活代客模式，请输入四位数字密码，此模式会限制车速和部分功能的访问权限。",
    "日常充电建议将充电上限设置为 80% 到 90%，以延长电池的使用寿命。",
    "要启用行车记录仪功能，需要在手套箱的 USB 接口中插入格式化的 U 盘。",
    "车辆在长时间不用时，建议将充电器连接到车辆并设置充电上限为 50% 到 60%。",
]


def quantize_model(input_path: str, output_path: str):
    """加载 BF16 模型 → AWQ 量化 → 保存 INT4 模型"""
    print(f"Loading BF16 model from: {input_path}")
    t0 = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)

    # Load model with AutoAWQ
    model = AutoAWQForCausalLM.from_pretrained(
        input_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # AWQ 需要 float16
        device_map="cpu",
    )
    print(f"  Model loaded ({time.time() - t0:.0f}s)")

    # Quantize
    print(f"Quantizing with config: {QUANT_CONFIG}")
    t1 = time.time()
    model.quantize(
        tokenizer=tokenizer,
        quant_config=QUANT_CONFIG,
        calib_data=CALIBRATION_SAMPLES,
    )
    print(f"  Quantization done ({time.time() - t1:.0f}s)")

    # Save
    print(f"Saving INT4 model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_quantized(
        output_path,
        safetensors=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_path)

    # Save quantization config manually
    with open(os.path.join(output_path, "quantize_config.json"), "w") as f:
        json.dump(QUANT_CONFIG, f, indent=2)

    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
    )
    print(f"✓ INT4 model saved: {output_path}")
    print(f"  Size: {total_size / 1e9:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="AWQ quantize merged ORPO v4 model")
    parser.add_argument("--input", default=MERGED_BF16_PATH, help="BF16 model path")
    parser.add_argument("--output", default=OUTPUT_INT4_PATH, help="INT4 output path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ BF16 model not found: {args.input}")
        print("   Run scripts/merge_orpo_v4.py first.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ 未检测到 GPU。AWQ 量化需要 CUDA。")
        print("   请在 RTX 4090 机器上运行此脚本：")
        print(f"   cd /root/autodl-tmp/rag-server")
        print(f"   conda activate rag")
        print(f"   unset OMP_NUM_THREADS")
        print(f"   python3 scripts/quantize_orpo_v4.py")
        sys.exit(1)

    print("=" * 60)
    print("AWQ INT4 Quantization")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()

    quantize_model(args.input, args.output)


if __name__ == "__main__":
    main()
