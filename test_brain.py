#!/usr/bin/env python3
"""
OpenTrade 大脑系统测试

演示多 Agent 协作决策和自主进化
"""

import sys
sys.path.insert(0, "/root/opentrade")

from opentrade.agents.brain import get_brain, analyze_market, run_evolution
from opentrade.agents.identity import get_agent_team
from opentrade.data.history_manager import get_history_manager, init_sample_data


def main():
    print("=" * 60)
    print("🧠 OpenTrade 自主进化系统演示")
    print("=" * 60)
    
    # 1. 初始化示例数据
    print("\n1️⃣ 初始化历史数据...")
    init_sample_data()
    
    # 2. 获取 Agent 团队
    print("\n2️⃣ Agent 团队:")
    team = get_agent_team()
    for role, agent in team.agents.items():
        identity = agent["identity"]
        print(f"   {identity.color} {identity.name} ({identity.role})")
        print(f"      专长: {', '.join(identity.expertise[:3])}")
    
    # 3. 获取大脑
    print("\n3️⃣ 启动大脑...")
    brain = get_brain()
    
    # 4. 市场分析演示
    print("\n4️⃣ 市场分析演示:")
    
    market_scenarios = [
        {
            "symbol": "BTC/USDT",
            "price": 67000,
            "fear_greed_index": 8,  # 极度恐惧
            "trend": "bearish",
            "volatility": 0.03,
        },
        {
            "symbol": "BTC/USDT",
            "price": 72000,
            "fear_greed_index": 72,  # 贪婪
            "trend": "bullish",
            "volatility": 0.025,
        },
        {
            "symbol": "BTC/USDT",
            "price": 68000,
            "fear_greed_index": 45,  # 中性
            "trend": "neutral",
            "volatility": 0.02,
        },
    ]
    
    for i, scenario in enumerate(market_scenarios, 1):
        print(f"\n   场景 {i}: 恐惧指数 {scenario['fear_greed_index']}")
        decision = analyze_market(scenario)
        entry_str = f"${decision.entry_price:,.0f}" if decision.entry_price else "N/A"
        print(f"   决策: {decision.action.upper()} {entry_str}")
        print(f"   置信度: {decision.confidence:.0%}")
    
    # 5. 生成 AI 上下文
    print("\n5️⃣ AI 上下文 (精选数据):")
    context = brain.generate_ai_context()
    print(f"   上下文长度: {len(context)} 字符")
    print(f"   保存路径: /root/opentrade/data/evolution/ai_context.md")
    
    with open("/root/opentrade/data/evolution/ai_context.md", "w") as f:
        f.write(context)
    print("   ✅ 已保存")
    
    # 6. 系统报告
    print("\n6️⃣ 系统状态:")
    brain.print_status()
    
    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)
    print("\n下一步:")
    print("- 使用真实 API 数据进行更多决策")
    print("- 执行进化以优化策略")
    print("- 记录交易结果以改进模型")


if __name__ == "__main__":
    main()
