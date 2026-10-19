import os
import asyncio
from nicegui import ui
from openai import OpenAI

# =========================================================
# 生命食谱 Life Cookbook
# 支持：DeepSeek + 智谱（OpenAI 兼容接口）
# 通义千问：暂未接入
#
# 环境变量：
#   DEEPSEEK_API_KEY=你的 DeepSeek API Key
#   ZHIPU_API_KEY=你的智谱 API Key
#   ZHIPU_MODEL=glm-5.3-flash  （按智谱控制台可用模型修改）
# =========================================================

SYSTEM_PROMPT = """
你是“生命食谱（Life Cookbook）”的 AI 生活方式助手。
请用中文提供清晰、谨慎、易理解的饮食、运动、睡眠、
兴趣体验、休息和心理放松建议。

生命食谱中的“食谱”不只指食物，也包括运动、睡眠、
电影、音乐、阅读、社交等能够丰富生活体验的活动。
尊重用户自主选择，不把生活安排变成强制打卡。
每个人都有选择自己人生的权利，没有一种生活方式适合所有人。
你的职责不是替用户决定应该怎样生活，而是帮助用户了解更多可能，
提供可选择、可调整的建议，并尊重用户接受、修改或拒绝建议的权利。
不要把健康、效率、成长或完成任务的数量当作衡量一个人价值的唯一标准。
用户不必为了符合某种理想标准而生活，也不应因没有完成建议而受到责备。
不要编造用户的体征、睡眠或健康数据，不进行疾病诊断。
不要建议极端节食或不安全的活动。
涉及严重、持续或紧急的健康问题时，建议向可信赖的成年人
或合格的医疗专业人员寻求帮助。
"""

# =========================================================
# 全局状态
# =========================================================

state = {
    "user_name": "Alex",
    "membership_tier": "👑 个人尊享 VIP 版 (全功能解锁)",
    "model_provider": "DeepSeek-R1 (核心生活医学推理)",
    "chat_messages": [
        {
            "role": "assistant",
            "content": (
                "你好！我是生命食谱的 AI 生活方式助手。\n\n"
                "你可以和我讨论饮食、运动、睡眠、休息，"
                "也可以探索电影、音乐、阅读等滋养精神生活的体验。"
                "你可以自主选择，不必把每一天变成必须完成的清单。"
            ),
        }
    ],
}

# =========================================================
# AI 调用逻辑
# =========================================================

def get_provider_config(provider):
    """根据界面选择返回 API 配置。"""

    if provider.startswith("DeepSeek"):
        return {
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": "https://api.deepseek.com",
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "env_name": "DEEPSEEK_API_KEY",
        }

    if provider.startswith("智谱"):
        return {
            "api_key": os.getenv("ZHIPU_API_KEY"),
            "base_url": "https://open.bigmodel.cn/api/paas/v4/",
            "model": os.getenv("ZHIPU_MODEL", "glm-5.3-flash"),
            "env_name": "ZHIPU_API_KEY",
        }

    raise ValueError("该模型暂未接入，请选择 DeepSeek 或智谱。")


def call_ai(provider, history):
    """
    同步调用模型。
    此函数由 asyncio.to_thread 在线程中执行，避免阻塞 NiceGUI。
    """

    config = get_provider_config(provider)

    if not config["api_key"]:
        raise ValueError(
            f"没有读取到环境变量 {config['env_name']}。"
            "请先在启动程序的终端或 IDE 运行配置中设置 API Key。"
        )

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=120.0,
        max_retries=2,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        max_tokens=4000,
        stream=False,
    )

    # 输出必要诊断信息，不输出 API Key。
    usage = getattr(response, "usage", None)
    if usage is not None:
        usage_info = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    else:
        usage_info = None

    if not response.choices:
        print(
            "[AI 调试] 没有 choices；"
            f"model={getattr(response, 'model', None)}, "
            f"usage={usage_info}"
        )
        raise RuntimeError("API 返回结果中没有 choices。")

    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)
    finish_reason = getattr(choice, "finish_reason", None)

    print(
        "[AI 调试] "
        f"provider={provider}, "
        f"requested_model={config['model']}, "
        f"returned_model={getattr(response, 'model', None)}, "
        f"finish_reason={finish_reason}, "
        f"usage={usage_info}, "
        f"content_empty={not (isinstance(content, str) and content.strip())}, "
        f"reasoning_present={bool(isinstance(reasoning, str) and reasoning.strip())}"
    )

    if isinstance(content, str) and content.strip():
        return content.strip()

    # 不将内部推理文本当成最终回答展示，只用来诊断。
    if isinstance(reasoning, str) and reasoning.strip():
        raise RuntimeError(
            "模型返回了 reasoning_content，但最终 content 为空。"
            f"finish_reason={finish_reason}。"
            "请检查模型的推理模式、模型 ID、输出限制及账户调用记录。"
        )

    if finish_reason == "length":
        raise RuntimeError(
            "模型输出达到长度限制，但没有返回正文。"
            "请缩短问题或检查模型输出配置。"
        )

    raise RuntimeError(
        "模型没有返回可显示的正文。"
        f"finish_reason={finish_reason}。"
        "请检查模型 ID、账户额度和智谱控制台的调用记录。"
    )


# =========================================================
# =========================================================
# 生命食谱：多因素加权匹配算法 V1
# 用于计算推荐类别与用户输入之间的规则匹配分
# =========================================================

CATEGORY_INTERESTS = {
    "营养": ["做饭", "饮食", "食物", "烹饪", "营养"],
    "活力": ["散步", "运动", "亲近自然", "骑行", "户外"],
    "修复": ["放松休息", "休息", "音乐", "阅读", "亲近自然"],
    "心灵": [
        "音乐", "阅读", "看电影", "绘画与创作",
        "与朋友聊天", "拼乐高", "研究机甲", "摄影", "创作"
    ],
}

MOOD_CATEGORY_SCORES = {
    "轻松": {
        "营养": 15, "活力": 15, "修复": 20, "心灵": 25
    },
    "平静": {
        "营养": 15, "活力": 15, "修复": 20, "心灵": 20
    },
    "有些疲惫": {
        "营养": 15, "活力": 5, "修复": 25, "心灵": 20
    },
    "精力充沛": {
        "营养": 15, "活力": 25, "修复": 10, "心灵": 15
    },
}


def calculate_match_score(category, preference, mood, interests, time_text):
    """返回规则匹配分（0—100）和可解释的推荐依据。"""
    score = 0
    reasons = []

    # 1. 用户优先关注类别：最高 25 分
    if preference == category:
        score += 25
        reasons.append("符合优先关注类别")
    elif preference == "不限":
        score += 15
        reasons.append("按综合偏好推荐")
    else:
        score += 5

    # 2. 当前状态与类别的适配：最高 25 分
    mood_score = MOOD_CATEGORY_SCORES.get(
        mood, {}
    ).get(category, 15)
    score += mood_score
    reasons.append("状态适配参考分：" + str(mood_score))

    # 3. 兴趣匹配：最高 35 分
    interest_text = interests or "未指定"
    matched = [
        keyword
        for keyword in CATEGORY_INTERESTS.get(category, [])
        if keyword in interest_text
    ]

    if matched:
        score += 35
        reasons.append("关联兴趣：" + "、".join(matched[:3]))
    elif interest_text == "未指定":
        score += 15
        reasons.append("未指定兴趣，采用通用建议")
    else:
        score += 5
        reasons.append("将结合用户兴趣进一步生成")

    # 4. 可用时间：最高 15 分
    try:
        minutes = int(time_text.split()[0])
    except (ValueError, AttributeError, IndexError):
        minutes = 20

    if minutes >= 20:
        score += 15
        reasons.append("时间条件已纳入规划")
    else:
        score += 10
        reasons.append("优先考虑短时活动")

    return min(score, 100), "；".join(reasons)
def get_category_interests(category, interests):
    """根据类别筛选更适合的用户兴趣。"""
    if not interests or interests == "未指定":
        return "未指定"

    all_interests = [
        item.strip()
        for item in interests.replace("，", "、").split("、")
        if item.strip()
    ]

    matched = [
        item for item in all_interests
        if item in CATEGORY_INTERESTS.get(category, [])
    ]

    # 只传递与当前类别直接相关的兴趣
    if matched:
        return "、".join(matched)

    # 没有直接匹配的兴趣时，让 AI 以类别目标为主
    return "未指定"
# ==================== 生命食谱：理念与文字档案 ====================

LIFE_COOKBOOK_PHILOSOPHY = {
    "核心理念": (
        "生命食谱不是替每个人规定生活应该是什么样子，"
        "而是帮助每个人找到属于自己的人生菜单。"
        "菜单不需要多么华丽、轰轰烈烈，适合自己就好。"
        "无论是爆辣入口、椒麻鲜香、浓油赤酱、老汤吊鲜，"
        "还是粗茶淡饭，你的人生亦由你自己决定口味。"
    ),

    "人生感悟": (
        "所有人都不可能完美，你、我，大家亦然。"
        "生命中有太多的可能，交织成独一无二的你。"
        "没有哪一种滋味必须被所有人认可，"
        "也没有哪一份人生菜单需要做到尽善尽美。"
        "去感受，去经历，去选择，去成为你自己。"
    ),

    "善良与边界": (
        "有时真正的善良是不必对他人负责到底，"
        "因为他们有自己的路要走；"
        "也不必对自己过于严苛，"
        "因为你已经做得很棒了。"
    ),

    "四季诗句": (
        "春天青风温如水，"
        "夏天淡阳依如木，"
        "秋天褐叶悯如土，"
        "冬天盛雪动如天"
    ),

    "星空寄语": (
        "世界没那么完美，但请抬头看看星空。"
        "你我都属于星星的孩子。"
        "不必缅怀，不必憎恶，"
        "不必强求感激，也不必深恋不放。"
        "愿你在有限的岁月里，尝到自己喜欢的滋味，"
        "见到自己想看的风景，也能在疲惫时安心歇一歇。"
        "生命无常，愿你安好。"
    ),

    "AI行为原则": [
        "人生菜单由用户自己决定，AI提供建议而不替用户做主。",
        "尊重用户的自主选择，不强迫用户完成每日清单。",
        "不以健康分数、匹配分数或完成次数定义一个人的价值。",
        "结合用户的状态、兴趣和可用时间提供建议，不强行套用兴趣。",
        "用户可以休息、改变计划，也可以暂时什么都不做。",
        "尊重人与人之间的边界，不要求用户为所有人的选择负责到底。",
        "回应用户时保持尊重，不因用户暂时没有行动而指责或羞辱用户。",
        "生活建议应当务实、适度，不把普通建议包装成诊断或治疗结论。",
    ],
}

# ================================================================
# 页面样式
# =========================================================

ui.query("body").style(
    "background-color: #F8FAFC; margin: 0; padding: 0; "
    'font-family: -apple-system, BlinkMacSystemFont, '
    '"Segoe UI", "Microsoft YaHei", sans-serif;'
)


# =========================================================
# 页面布局
# =========================================================

with ui.column().classes("w-full min-h-screen p-6 gap-6"):

    # 顶栏
    with ui.row().classes(
        "w-full items-center justify-between bg-white px-6 py-4 "
        "shadow-sm rounded-2xl border border-slate-100"
    ):
        with ui.row().classes("items-center gap-3"):
            ui.label("🧬 生命食谱 (Life Cookbook)").classes(
                "text-lg font-bold text-slate-800"
            )
            tier_chip = ui.chip(
                state["membership_tier"],
                color="blue-50",
                text_color="blue-700",
            ).classes("text-xs font-semibold")

        with ui.row().classes("items-center gap-4"):
            ui.label(
                f"欢迎回来，{state['user_name']}"
            ).classes("text-sm text-slate-600 font-medium")
            ui.badge("原型演示版", color="green").classes("text-xs")

    # 主体双栏
    with ui.row().classes("w-full items-start gap-6 flex-nowrap"):

        # 左侧设置
        with ui.card().classes(
            "w-[320px] shrink-0 p-5 bg-white shadow-sm rounded-2xl "
            "border border-slate-100 gap-4"
        ):
            ui.label("⚙️ 个人中心与模型调度").classes(
                "text-sm font-bold text-slate-800"
            )

            ui.label("会员界面选项（原型展示）").classes(
                "text-xs text-slate-400 font-semibold"
            )
            tier_sel = ui.select(
                [
                    "🆓 免费体验版 (基础功能)",
                    "🔑 极客自定义版 (BYOK 模式)",
                    "👑 个人尊享 VIP 版 (全功能解锁)",
                ],
                value=state["membership_tier"],
            ).classes("w-full")

            ui.label("🤖 AI 模型切换").classes(
                "text-xs text-slate-400 font-semibold"
            )
            model_options = [
                "DeepSeek-R1 (核心生活医学推理)",
                "智谱 GLM (生活食谱助手)",
                "通义千问-Max (尚未接入)",
            ]
            model_sel = ui.select(
                model_options,
                value=state["model_provider"],
            ).classes("w-full")

            def handle_config_change():
                state["membership_tier"] = tier_sel.value
                state["model_provider"] = model_sel.value
                tier_chip.text = tier_sel.value

                if model_sel.value.startswith("通义千问"):
                    ui.notify(
                        "通义千问目前尚未接入，请选择 DeepSeek 或智谱。",
                        type="warning",
                    )
                else:
                    ui.notify(
                        f"已选择模型：{model_sel.value}",
                        type="positive",
                    )

            tier_sel.on(
                "update:model-value",
                lambda _: handle_config_change(),
            )
            model_sel.on(
                "update:model-value",
                lambda _: handle_config_change(),
            )

            ui.separator()
            ui.label(
                "生命食谱不仅关注饮食，也关注运动、睡眠、休息、"
                "电影、音乐、阅读和社交等生活体验。"
            ).classes("text-xs text-slate-400 leading-relaxed")

            with ui.expansion("API 配置检查", icon="key").classes("w-full"):
                def check_api_config():
                    provider = state["model_provider"]
                    if provider.startswith("DeepSeek"):
                        key_exists = bool(os.getenv("DEEPSEEK_API_KEY"))
                        model_name = os.getenv(
                            "DEEPSEEK_MODEL", "deepseek-chat"
                        )
                        env_name = "DEEPSEEK_API_KEY"
                    elif provider.startswith("智谱"):
                        key_exists = bool(os.getenv("ZHIPU_API_KEY"))
                        model_name = os.getenv(
                            "ZHIPU_MODEL", "glm-5.3-flash"
                        )
                        env_name = "ZHIPU_API_KEY"
                    else:
                        ui.notify("当前模型尚未接入。", type="warning")
                        return

                    if key_exists:
                        ui.notify(
                            f"已读取 {env_name}；当前模型 ID：{model_name}。"
                            "这只检查环境变量是否存在，不代表 API 调用已成功。",
                            type="positive",
                        )
                    else:
                        ui.notify(
                            f"未读取到 {env_name}，请检查环境变量。",
                            type="warning",
                        )

                ui.button(
                    "检查当前 API 配置",
                    on_click=check_api_config,
                ).classes("w-full")

            ui.separator()

            ui.label("🌱 生命食谱的初心").classes(
                "text-sm font-bold text-slate-800"
            )

            ui.label(
                "每个人，都有选择自己人生的权利。"
            ).classes(
                "text-sm font-semibold text-slate-700 leading-6"
            )

            ui.label(
                "我们不替你决定该怎样生活，"
                "也不要求你成为别人眼中的理想模样。"
                "生命食谱希望借助 AI，"
                "陪你探索更多可能，"
                "而人生的方向，始终由你自己选择。"
            ).classes(
                "text-xs text-slate-500 leading-6"
            )

            ui.label(
                "生命无常，愿你安好。"
            ).classes(
                "text-sm text-slate-600 leading-6"
            )

        # 右侧工作区
        with ui.column().classes("flex-1 gap-6 min-w-0"):

            # 指标卡：原型数据明确标注，避免误认为真实健康测量
            with ui.row().classes("w-full grid grid-cols-4 gap-4"):
                with ui.card().classes(
                    "p-4 bg-white shadow-sm rounded-2xl "
                    "border border-slate-100"
                ):
                    ui.label("综合健康评分").classes(
                        "text-xs text-slate-400"
                    )
                    ui.label("演示数据").classes(
                        "text-lg font-extrabold text-emerald-600"
                    )

                with ui.card().classes(
                    "p-4 bg-white shadow-sm rounded-2xl "
                    "border border-slate-100"
                ):
                    ui.label("当前绑定引擎").classes(
                        "text-xs text-slate-400"
                    )
                    ui.label().bind_text_from(
                        state,
                        "model_provider",
                        backward=lambda p: p.split(" ")[0],
                    ).classes("text-lg font-extrabold text-blue-600")

                with ui.card().classes(
                    "p-4 bg-white shadow-sm rounded-2xl "
                    "border border-slate-100"
                ):
                    ui.label("方案匹配度").classes(
                        "text-xs text-slate-400"
                    )
                    ui.label("待接入真实算法").classes(
                        "text-sm font-bold text-slate-800"
                    )

                with ui.card().classes(
                    "p-4 bg-white shadow-sm rounded-2xl "
                    "border border-slate-100"
                ):
                    ui.label("连续记录天数").classes(
                        "text-xs text-slate-400"
                    )
                    ui.label("待接入记录").classes(
                        "text-sm font-bold text-blue-600"
                    )

            # 功能选项卡
            with ui.tabs().classes(
                "w-full bg-white rounded-xl p-2 shadow-sm "
                "border border-slate-100 text-slate-700"
            ) as tabs:
                tc = ui.tab("💬 生命对话")
                t1 = ui.tab("🩺 健康解读")
                t2 = ui.tab("🍃 人生菜单")
                t3 = ui.tab("📈 成长回响")

            with ui.tab_panels(tabs, value=tc).classes(
                "w-full bg-transparent"
            ):

                # AI 对话
                with ui.tab_panel(tc):
                    with ui.card().classes(
                        "p-6 bg-white shadow-sm rounded-2xl "
                        "border border-slate-100 w-full gap-4"
                    ):
                        ui.label("🤖 生命对话").classes(
                            "text-base font-bold text-slate-800"
                        )
                        ui.label("在交流中梳理想法，找到适合自己的生活方式").classes(
                            "text-sm text-slate-500"
                        )

                        ui.label().bind_text_from(
                            state,
                            "model_provider",
                            backward=lambda p: f"当前选择：{p}",
                        ).classes("text-xs text-slate-400 mb-2")

                        c_box = ui.column().classes(
                            "w-full h-[380px] overflow-y-auto p-4 "
                            "bg-slate-50 rounded-xl gap-3 "
                            "border border-slate-100"
                        )

                        def show_msgs():
                            c_box.clear()
                            with c_box:
                                for item in state["chat_messages"]:
                                    if item["role"] == "user":
                                        with ui.row().classes(
                                            "w-full justify-end"
                                        ):
                                            ui.label(
                                                item["content"]
                                            ).classes(
                                                "bg-blue-600 text-white "
                                                "px-4 py-2 rounded-2xl "
                                                "max-w-md text-sm whitespace-pre-wrap"
                                            )
                                    else:
                                        with ui.row().classes(
                                            "w-full justify-start"
                                        ):
                                            ui.label(
                                                item["content"]
                                            ).classes(
                                                "bg-white text-slate-800 "
                                                "px-4 py-2 rounded-2xl "
                                                "max-w-md text-sm shadow-sm "
                                                "border border-slate-100 "
                                                "whitespace-pre-wrap"
                                            )

                        show_msgs()

                        with ui.row().classes(
                            "w-full gap-3 items-center"
                        ):
                            in_box = ui.input(
                                placeholder=(
                                    "聊聊饮食、运动、睡眠、电影、音乐或今天的生活安排……"
                                )
                            ).classes("flex-1")

                            async def send():
                                if not in_box.value or not in_box.value.strip():
                                    return

                                provider = state["model_provider"]

                                if provider.startswith("通义千问"):
                                    ui.notify(
                                        "通义千问尚未接入，请选择 DeepSeek 或智谱。",
                                        type="warning",
                                    )
                                    return

                                txt = in_box.value.strip()
                                in_box.value = ""

                                state["chat_messages"].append({
                                    "role": "user",
                                    "content": txt,
                                })
                                state["chat_messages"].append({
                                    "role": "assistant",
                                    "content": (
                                        f"正在连接 {provider.split()[0]}，请稍候……"
                                    ),
                                })
                                show_msgs()

                                # 排除最后一条“正在连接……”临时消息
                                history = [
                                    {
                                        "role": msg["role"],
                                        "content": msg["content"],
                                    }
                                    for msg in state["chat_messages"][:-1][-12:]
                                    if msg["role"] in ("user", "assistant")
                                ]

                                try:
                                    answer = await asyncio.to_thread(
                                        call_ai, provider, history
                                    )
                                    state["chat_messages"][-1] = {
                                        "role": "assistant",
                                        "content": answer,
                                    }

                                except Exception as exc:
                                    print(
                                        f"[AI 调用失败] {provider}: {exc!r}"
                                    )
                                    state["chat_messages"][-1] = {
                                        "role": "assistant",
                                        "content": (
                                            "这次没有成功取得 AI 回复。\n"
                                            f"原因：{str(exc)}\n\n"
                                            "请查看启动程序的终端日志，确认模型 ID、"
                                            "API Key、账户额度和网络连接。"
                                            "不要把 API Key 发到聊天窗口或公开日志中。"
                                        ),
                                    }

                                show_msgs()

                            ui.button(
                                "发送",
                                on_click=send,
                            ).classes(
                                "bg-blue-600 text-white px-6 py-2.5 rounded-xl"
                            )

                # 体检报告解析演示模块
                with ui.tab_panel(t1):
                    with ui.card().classes(
                        "p-6 bg-white shadow-sm rounded-2xl "
                        "border border-slate-100 w-full gap-4"
                    ):
                        ui.label("🩺 健康解读").classes(
                            "text-xl font-bold text-slate-800"
                        )
                        ui.label("读懂身体的信笺，了解报告中的指标与含义").classes(
                            "text-sm text-slate-500"
                        )

                        ui.label(
                            "此模块目前只是界面演示，上传文件不会自动完成识别。"
                            "请勿上传真实敏感健康资料用于未验证的功能。"
                        ).classes("text-sm text-slate-500")
                        ui.upload(
                            label="选择文件（当前仅为界面演示）",
                            auto_upload=False,
                        ).classes("w-full")
                        ui.button(
                            "检查解析模块状态",
                            on_click=lambda: ui.notify(
                                "报告解析功能尚未接入，当前不会生成真实分析结果。",
                                type="warning",
                            ),
                        ).classes(
                            "bg-blue-600 text-white font-medium px-6 py-2.5 rounded-xl"
                        )


                # 生活方式食谱演示模块
                with ui.tab_panel(t2):
                    recipe_categories = ["营养", "活力", "修复", "心灵"]
                    recipe_texts = {}
                    recipe_match_labels = {}
                    recipe_status = None

                    with ui.card().classes("w-full p-5"):
                        ui.label("🍃 人生菜单").classes(
                            "text-xl font-bold text-slate-800"
                        )

                        ui.label(
                            "人生不只有一种活法，生活也不只有一种滋味。"
                            "有人喜欢爆辣入口，有人偏爱椒麻鲜香；"
                            "有人钟情浓油赤酱，有人满足于粗茶淡饭。"
                            "没有哪一种人生必须成为标准答案，"
                            "适合自己的，就是属于自己的菜单。"
                        ).classes("text-sm text-slate-600 leading-7")

                        ui.label(
                            "根据今天的心情、兴趣与可用时间，"
                            "为营养、活力、修复与心灵，调配属于今天的生活建议。"
                        ).classes("text-sm text-slate-500 leading-6")

                        with ui.row().classes("w-full items-end gap-4"):
                            preferred_category = ui.select(
                                ["不限", "营养", "活力", "修复", "心灵"],
                                value="不限",
                                label="今天想优先关注什么？",
                            ).classes("flex-1")

                            available_time = ui.select(
                                ["10 分钟", "20 分钟", "30 分钟", "60 分钟"],
                                value="20 分钟",
                                label="今天可用的时间",
                            ).classes("flex-1")
                            with ui.row().classes("w-full gap-4"):
                                today_mood = ui.select(
                                    ["轻松", "平静", "有些疲惫", "精力充沛"],
                                    value="平静",
                                    label="今天的状态",
                                ).classes("flex-1")

                                # 预设兴趣：用户可以自由多选
                                activity_options = [
                                    "音乐", "阅读", "看电影", "散步", "运动",
                                    "绘画与创作", "与朋友聊天", "亲近自然",
                                    "做饭", "放松休息",
                                ]

                                activity_checks = {}

                                with ui.column().classes("w-full gap-2"):
                                    ui.label("感兴趣的活动（可多选）").classes(
                                        "font-bold text-gray-800"
                                    )
                                    ui.label(
                                        "选择你喜欢的活动，也可以添加自己的兴趣。"
                                    ).classes("text-sm text-gray-500")

                                    with ui.grid(columns=2).classes("w-full gap-2"):
                                        for activity in activity_options:
                                            activity_checks[activity] = ui.checkbox(
                                                activity, value=activity in ["音乐", "阅读", "看电影"]
                                            )

                                    # 自定义兴趣列表
                                    custom_activities = []
                                    custom_activity_input = ui.input(
                                        label="添加自己的兴趣",
                                        placeholder="例如：拼乐高、研究机甲、摄影",
                                    ).classes("w-full")

                                    custom_activity_list = ui.column().classes("w-full gap-1")


                                    def refresh_custom_activities():
                                        custom_activity_list.clear()
                                        with custom_activity_list:
                                            for index, activity in enumerate(custom_activities):
                                                with ui.row().classes("items-center gap-2"):
                                                    ui.label(f"• {activity}").classes("text-gray-700")
                                                    ui.button(
                                                        "删除",
                                                        on_click=lambda i=index: remove_custom_activity(i),
                                                    ).props("flat dense color=negative")


                                    def add_custom_activity():
                                        activity = custom_activity_input.value.strip()

                                        if not activity:
                                            ui.notify("请先输入一个活动名称", type="warning")
                                            return

                                        existing = [
                                            name for name, checkbox in activity_checks.items()
                                            if checkbox.value
                                        ]

                                        if activity in existing or activity in custom_activities:
                                            ui.notify("这个活动已经添加过了", type="warning")
                                            return

                                        custom_activities.append(activity)
                                        custom_activity_input.value = ""
                                        refresh_custom_activities()


                                    def remove_custom_activity(index):
                                        if 0 <= index < len(custom_activities):
                                            custom_activities.pop(index)
                                            refresh_custom_activities()


                                    ui.button(
                                        "＋ 添加活动", on_click=add_custom_activity
                                    ).props("outline")


                            def get_today_interests():
                                selected = [
                                    activity
                                    for activity, checkbox in activity_checks.items()
                                    if checkbox.value
                                ]

                                selected.extend(
                                    activity.strip()
                                    for activity in custom_activities
                                    if isinstance(activity, str) and activity.strip()
                                )

                                # 去除重复兴趣，同时保留用户选择的顺序
                                selected = list(dict.fromkeys(selected))

                                return "、".join(selected) if selected else "未指定"






                        with ui.row().classes("w-full items-center"):
                            generate_button = ui.button(
                                "✨ 调配今日菜单"
                            ).classes(
                                "bg-blue-600 text-white font-bold px-6 py-3 rounded-xl"
                            )

                            status_label = ui.label(
                                "尚未生成。点击按钮开始。"
                            ).classes("text-gray-500")

                            ui.separator()

                            ui.label(
                                "菜单是为你提供的可能，不是必须完成的任务。"
                                "你可以自由选择、修改或跳过，按自己的方式生活。"
                            ).classes(
                                "w-full text-sm text-gray-500 leading-6"
                            )

                            with ui.grid(columns=2).classes("w-full gap-4"):
                             for category in recipe_categories:
                                with ui.card().classes(
                                        "w-full p-4 border border-gray-200 rounded-xl"
                                ):
                                    ui.label(
                                        {
                                            "营养": "🥗 营养",
                                            "活力": "🏃 活力",
                                            "修复": "🌙 修复",
                                            "心灵": "🎨 心灵",
                                        }[category]
                                    ).classes("text-xl font-bold")

                                    recipe_match_labels[category] = ui.label(
                                        "推荐匹配参考：等待计算……"
                                    ).classes("text-xs text-blue-600")

                                    recipe_texts[category] = ui.label(
                                        "等待生成个性化建议……"
                                    ).classes("text-gray-700 whitespace-pre-wrap")

                                    ui.button(
                                        "🔄 单独换一个建议",
                                        on_click=lambda c=category: regenerate_one(c),
                                    ).props("outline").classes("mt-2")


                    async def request_recipe(category, time_text, preference, mood, interests):
                        recent_feedback = [
                                              record
                                              for record in feedback_records
                                              if record.get("category") == category
                                          ][-5:]

                        if recent_feedback:
                            feedback_context = "\n".join(
                                f"- 用户反馈：{record.get('choice')}；"
                                f"体验评分：{record.get('score')}/5"
                                for record in recent_feedback
                            )
                        else:
                            feedback_context = "暂无该类别的历史反馈。"
                        prompt = f"""
                    请为“生命食谱”生成一份简洁、具体、可执行的生活方式建议。

                    【用户信息】
                    【该类别的历史反馈】
                    {feedback_context}

                    请参考历史反馈调整建议：
                    1. 对“不太适合”或低评分的反馈，尽量调整建议的形式或内容。
                    2. 对“适合我”或高评分的反馈，可以保留相似的体验特点，但不要机械重复。
                    3. “想换一个”表示用户希望获得替代方案，不代表用户永远排斥该类别。
                    4. 历史反馈仅供参考，优先尊重用户本次填写的心情、兴趣与可用时间。
                    5. 不要在回答中复述这些反馈记录。
                    可用时间：{time_text}
                    优先关注类别：{preference}
                    本次生成类别：{category}
                    当前心情：{mood}
                    兴趣活动：{interests}
                    【算法推荐依据】
                    匹配分数：{calculate_match_score(
                        category, preference, mood, interests, time_text
                    )[0]} / 100
                    推荐理由：{calculate_match_score(
                        category, preference, mood, interests, time_text
                    )[1]}
                    
                    请将算法推荐依据作为辅助参考，而不是硬性要求。
                    优先保证本类别的核心目标，再结合兴趣进行个性化调整。
                    只有当兴趣与本类别自然相关时，才将其转化为具体行动；
                    如果关联较弱，可以不使用该兴趣，不要为了迎合兴趣而生硬套用。
                    
                    【项目理念】
                    “食谱”不仅指食物，也包括运动、休息、睡眠习惯、音乐、
                    阅读、影视、社交和兴趣活动等日常生活安排。
                    
                    【生命食谱的生活理念】

你不仅要提供生活建议，也要尊重每个人选择自己人生菜单的权利。

1. 尊重用户的自主选择。建议是邀请，不是命令；不要求用户完成每日清单，也不因用户拒绝建议而指责对方。

2. 不以健康分数、匹配分数、任务完成次数衡量一个人的价值。算法只用于辅助推荐，不代表对用户本人的评价。

3. 善良需要尊重边界。有时不必对他人负责到底，因为他们有自己的路要走；也不必对自己过于严苛，因为人可以休息、改变计划，也可以暂时什么都不做。

4. 根据用户的心情、可用时间和兴趣提供切实可行的建议。兴趣与当前类别关联不大时，不要强行加入，也不要为了体现个性而重复提及同一兴趣。

5. 允许生活有不同的滋味。建议可以简单、平凡、安静，不必刻意追求高效、完美或轰轰烈烈。

这些理念应自然地体现在建议内容与表达方式中，不要逐条复述本规则，不要输出说教式的品牌宣言。
                    
                    
                    
                    
                    
                    【生成要求】
                    1. 只生成“{category}”类别的建议，不要混入其他类别。
                    2. 必须考虑用户的可用时间、当前心情和兴趣活动。
                    3. 兴趣活动是个性化建议的重要依据，不能只在文字中提及，
                       必须尽可能将其转化为具体行动。
                    4. 如果用户喜欢拼乐高，可以给出具体的搭建小任务；
                       如果喜欢音乐，可以安排听歌或音乐探索；
                       如果喜欢阅读，可以安排阅读一小段内容。
                    5. 如果有自定义兴趣，优先考虑该兴趣；如果与本类别关联较弱，
                       可以自然结合，不要生硬套用。
                    6. 如果兴趣为“未指定”，不要自行编造用户的兴趣。
                    7. 建议必须适合用户的可用时间，普通人能够实际执行。
                    8. 结合心情调整活动的氛围，但不要据此推断用户的健康状况。
                    9. 营养类别提供一般、均衡的饮食建议，不推荐极端节食；
                       活力类别优先考虑适度活动；
                       修复类别优先考虑休息和放松；
                       心灵类别优先考虑阅读、音乐、创作、兴趣或适度社交。
                    10. 不做医学诊断，不承诺治疗效果，不提供危险活动建议。

                    【输出格式】
                    只输出以下三项：
                    建议名称：用一句话概括。
                    具体行动：给出1至2个清晰步骤，至少有一个步骤尽可能体现用户的兴趣。
                    简短理由：用一句话说明这项建议的价值。

                    总长度控制在约100至150个汉字。
                    不要输出开场白、总结、JSON或推理过程。
                    不要重复用户输入来凑字数。
                    直接输出最终建议。
                            """
                        provider = state["model_provider"]
                        print(f"[食谱输入] 类别={category}，兴趣={interests}", flush=True)

                        answer = await asyncio.to_thread(
                            call_ai,
                            provider,
                            [{"role": "user", "content": prompt}],
                        )
                        return answer.strip()




                    async def regenerate_one(category):
                        status_label.text = f"正在重新生成「{category}」建议……"
                        try:
                            answer = await request_recipe(
                                category,
                                available_time.value,
                                preferred_category.value,
                                today_mood.value,
                                get_category_interests(category,
                                get_today_interests()
                             )
                            )
                            recipe_texts[category].text = answer

                            score, reason = calculate_match_score(
                                category,
                                preferred_category.value,
                                today_mood.value,
                                get_category_interests(
                                    category,
                                    get_today_interests()
                                ),
                                available_time.value
                            )
                            recipe_match_labels[category].text = (
                                f"推荐匹配参考：{score}/100\n"
                                f"推荐理由：{reason}"
                            )

                            status_label.text = f"「{category}」已更新。"
                        except Exception as e:
                            print(f"[食谱生成错误] {type(e).__name__}: {e}", flush=True)
                            status_label.text = f"生成失败：{e}"
                            ui.notify(
                                "生成失败，请检查 API 配置或网络后重试。",
                                type="negative",
                            )


                    async def generate_all():
                        generate_button.disable()
                        status_label.text = "AI 正在准备你的生活食谱，请稍候……"

                        try:
                            results = await asyncio.gather(
                                *[
                                    request_recipe(
                                        category,
                                        available_time.value,
                                        preferred_category.value,
                                        today_mood.value,
                                        get_category_interests(category,get_today_interests())
                                    )
                                    for category in recipe_categories
                                ]
                            )

                            for category, answer in zip(recipe_categories, results):
                                recipe_texts[category].text = answer

                                score, reason = calculate_match_score(
                                    category,
                                    preferred_category.value,
                                    today_mood.value,
                                    get_category_interests(
                                        category,
                                        get_today_interests()
                                    ),
                                    available_time.value
                                )
                                recipe_match_labels[category].text = (
                                    f"推荐匹配参考：{score}/100\n"
                                    f"推荐理由：{reason}"
                                )

                            status_label.text = "今日生活食谱已生成，可以单独替换任意类别。"

                        except Exception as e:
                            print(f"[食谱生成错误] {type(e).__name__}: {e}", flush=True)
                            status_label.text = f"生成失败：{e}"
                            ui.notify(
                                "食谱生成失败，请检查模型配置、API 额度和网络连接。",
                                type="negative",
                            )
                        finally:
                            generate_button.enable()


                    generate_button.on_click(generate_all)

                # 成长回响模块
                with ui.tab_panel(t3):
                    with ui.card().classes(
                            "p-6 bg-white shadow-sm rounded-2xl "
                            "border border-slate-100 w-full gap-4"
                    ):
                        ui.label("🌱 成长回响").classes(
                            "text-xl font-bold text-slate-800"
                        )

                        ui.label(
                            "你的感受，值得被认真对待。"
                        ).classes(
                            "text-base font-semibold text-slate-700"
                        )

                        ui.label(
                            "每个人的生活都有自己的滋味。"
                            "适合别人的建议，不一定适合你；"
                            "不喜欢，可以拒绝；想改变，也随时可以重新选择。"
                        ).classes(
                            "text-sm text-slate-500 leading-6"
                        )

                        ui.separator()

                        ui.label(
                            "成长不必每天都有进步。"
                        ).classes(
                            "text-sm font-semibold text-slate-700"
                        )

                        ui.label(
                            "有时向前走，有时停下来休息，也是一种生活。"
                            "没有完成计划，不代表今天的你做得不够好。"
                        ).classes(
                            "text-sm text-slate-500 leading-6"
                        )

                        ui.label(
                            "人生的方向，始终由你选择。"
                        ).classes(
                            "text-sm font-semibold text-slate-700"
                        )

                        ui.label(
                            "生命食谱希望陪你探索更多可能，"
                            "而不是替你决定应该怎样生活。"
                        ).classes(
                            "text-sm text-slate-500 leading-6"
                        )

                        ui.separator()
                        feedback_category = ui.select(
                            ["营养", "活力", "修复", "心灵"],
                            value="心灵",
                            label="这次反馈对应哪类建议",
                        ).classes("w-full")

                        ui.label(
                            "🌿 今天的生活建议，适合你吗？"
                        ).classes(
                            "text-base font-bold text-slate-800"
                        )

                        feedback_choice = ui.radio(
                            {
                                "适合我": "适合我",
                                "不太适合": "不太适合",
                                "想换一个": "想换一个",
                            },
                            value="适合我",
                        ).props("inline").classes("w-full")

                        ui.label(
                            "请为今天的建议打个分"
                        ).classes("text-sm text-slate-600")

                        feedback_score = ui.slider(
                            min=1,
                            max=5,
                            value=3,
                        ).props(
                            "label-always"
                        ).classes("w-full")

                        feedback_records = []


                        def submit_feedback():
                            record = {
                                "category": feedback_category.value,
                                "choice": feedback_choice.value,
                                "score": feedback_score.value,
                            }

                            feedback_records.append(record)

                            ui.notify(
                                "感谢你的真实反馈。"
                                "你的选择值得被尊重。",
                                type="positive",
                            )


                        ui.button(
                            "🌱 提交反馈",
                            on_click=submit_feedback,
                        ).classes(
                            "bg-blue-600 text-white font-medium "
                            "px-6 py-2.5 rounded-xl"
                        )

                        ui.label(
                            "你的反馈目前保存在本次运行的内存中。"
                            "程序重启后会清空，暂未接入数据库，"
                            "反馈是调整建议的参考，不是对你的固定判断。"
                        ).classes(
                            "text-xs text-slate-400 leading-5"
                        )

ui.run(
    port=8099,
    title="生命食谱 | Life Cookbook",
    favicon="🧬",
    reload=False,
)
