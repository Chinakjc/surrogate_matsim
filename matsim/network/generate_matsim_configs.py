#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  

import argparse  
import re  
from pathlib import Path  
import sys  

def replace_first_value(xml_text: str, param_name: str, new_value: str) -> str:  
    """  
    在 xml 文本中，将第一个 <param name="param_name" value="..."> 的 value 替换为 new_value。  
    使用非贪婪与捕获组，仅替换一次，不改变其他内容与格式。  
    """  
    pattern = re.compile(r'(<param\s+[^>]*\bname\s*=\s*"' + re.escape(param_name) + r'"\s+[^>]*\bvalue\s*=\s*")[^"]*(")',  
                         flags=re.IGNORECASE)  
    # 仅替换第一次出现  
    replaced, count = pattern.subn(r'\1' + new_value + r'\2', xml_text, count=1)  
    if count == 0:  
        raise RuntimeError(f'未在 config 中找到 param name="{param_name}"（或该 param 没有 value 属性）')  
    return replaced  

def main():  
    parser = argparse.ArgumentParser(  
        description="基于纯文本替换生成 MATSim 批量 config_n.xml（不解析 XML，最小改动）"  
    )  
    parser.add_argument("--config", required=True, help="原始可运行的 MATSim 配置文件路径")  
    parser.add_argument("--networks-dir", required=True, help="包含 network_scenario_n.xml.gz 的目录")  
    parser.add_argument("--output-configs-dir", required=True, help="输出 config_n.xml 的目录")  
    parser.add_argument("--network-prefix", default="networks_idf/scenario",  
                        help="写入到 config 的网络路径前缀（默认：networks_idf/scenario）")  
    args = parser.parse_args()  

    config_path = Path(args.config)  
    networks_dir = Path(args.networks_dir)  
    out_dir = Path(args.output_configs_dir)  
    net_prefix = args.network_prefix.rstrip("/")  

    if not config_path.is_file():  
        print(f"错误：找不到配置文件：{config_path}", file=sys.stderr)  
        sys.exit(1)  
    if not networks_dir.is_dir():  
        print(f"错误：找不到网络目录：{networks_dir}", file=sys.stderr)  
        sys.exit(1)  

    out_dir.mkdir(parents=True, exist_ok=True)  

    # 读取原始 config 文本（UTF-8，不改编码与 BOM）  
    src = config_path.read_text(encoding="utf-8")  

    # 列举网络文件，识别 n  
    pattern = re.compile(r"^network_scenario_(\d+)\.xml\.gz$")  
    items = []  
    for p in networks_dir.iterdir():  
        if p.is_file():  
            m = pattern.match(p.name)  
            if m:  
                n = int(m.group(1))  
                items.append((n, p.name))  

    if not items:  
        print(f"警告：目录 {networks_dir} 下未找到 network_scenario_n.xml.gz", file=sys.stderr)  
        sys.exit(0)  

    items.sort(key=lambda x: x[0])  

    for n, filename in items:  
        # 对每个 n 从原始文本重新开始，确保独立生成  
        text = src  

        # 替换 outputDirectory（第一个匹配）  
        text = replace_first_value(text, "outputDirectory", f"simulation_output/{n}")  

        # 替换 inputNetworkFile（第一个匹配）  
        text = replace_first_value(text, "inputNetworkFile", f"{net_prefix}/{filename}")  

        out_file = out_dir / f"config_{n}.xml"  
        out_file.write_text(text, encoding="utf-8")  
        print(f"生成：{out_file}")  

if __name__ == "__main__":  
    main()