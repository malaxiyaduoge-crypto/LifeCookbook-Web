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
# 页面样式
# =========================================================

ui.query("body").style(
    "background-color: #F8FAFC; margin: 0; padding: 0; "
    'font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", '
    "Roboto, sans-serif;"
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
                tc = ui.tab("💬 AI 智能对话")
                t1 = ui.tab("🩺 体检报告解析（演示）")
                t2 = ui.tab("🎯 生活方式食谱")
                t3 = ui.tab("📊 反馈追踪（演示）")

            with ui.tab_panels(tabs, value=tc).classes(
                "w-full bg-transparent"
            ):

                # AI 对话
                with ui.tab_panel(tc):
                    with ui.card().classes(
                        "p-6 bg-white shadow-sm rounded-2xl "
                        "border border-slate-100 w-full gap-4"
                    ):
                        ui.label("🤖 生命食谱 AI 助手").classes(
                            "text-base font-bold text-slate-800"
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
                        ui.label("🩺 体检报告解析").classes(
                            "text-base font-bold text-slate-800"
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
                    recipe_status = None

                    with ui.card().classes("w-full p-5"):
                        ui.label("🧬 生命食谱").classes(
                            "text-2xl font-bold text-gray-800"
                        )
                        ui.label(
                            "生活不只有饮食。让 AI 根据你的时间与偏好，"
                            "为营养、活力、修复和心灵生成可执行的生活建议。"
                        ).classes("text-gray-600")

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
                                    selected.extend(custom_activities)
                                    return "、".join(selected) if selected else "未指定"





                        with ui.row().classes("w-full items-center"):
                            generate_button = ui.button(
                                "✨ 生成今日生活食谱"
                            ).classes(
                                "bg-blue-600 text-white font-bold px-6 py-3 rounded-xl"
                            )

                            status_label = ui.label(
                                "尚未生成。点击按钮开始。"
                            ).classes("text-gray-500")

                        ui.separator()

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

                                    recipe_texts[category] = ui.label(
                                        "等待生成个性化建议……"
                                    ).classes("text-gray-700 whitespace-pre-wrap")

                                    ui.button(
                                        "🔄 单独换一个建议",
                                        on_click=lambda c=category: regenerate_one(c),
                                    ).props("outline").classes("mt-2")


                    async def request_recipe(category, time_text, preference, mood, interests):
                        prompt = f"""
                    请为“生命食谱”生成一份简洁、具体、可执行的生活方式建议。

                    【用户信息】
                    可用时间：{time_text}
                    优先关注类别：{preference}
                    本次生成类别：{category}
                    当前心情：{mood}
                    兴趣活动：{interests}

                    【项目理念】
                    “食谱”不仅指食物，也包括运动、休息、睡眠习惯、音乐、
                    阅读、影视、社交和兴趣活动等日常生活安排。

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
                                get_today_interests()
                            )
                            recipe_texts[category].text = answer
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
                                        get_today_interests()
                                    )
                                    for category in recipe_categories
                                ]
                            )

                            for category, answer in zip(recipe_categories, results):
                                recipe_texts[category].text = answer

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


                # 反馈追踪演示模块
                with ui.tab_panel(t3):
                    with ui.card().classes(
                        "p-6 bg-white shadow-sm rounded-2xl "
                        "border border-slate-100 w-full gap-4"
                    ):
                        ui.label("📊 反馈追踪").classes(
                            "text-base font-bold text-slate-800"
                        )
                        ui.label(
                            "当前尚未保存真实打卡记录。后续会支持完成、跳过、"
                            "不喜欢和自定义反馈，并用于改善推荐。"
                        ).classes("text-sm text-slate-500")
                        ui.slider(
                            min=0,
                            max=100,
                            value=50,
                        ).classes("w-full")
                        ui.button(
                            "查看开发状态",
                            on_click=lambda: ui.notify(
                                "反馈持久化功能尚未接入数据库。",
                                type="info",
                            ),
                        ).classes(
                            "bg-blue-600 text-white font-medium px-6 py-2.5 rounded-xl"
                        )


ui.run(
    host="0.0.0.0",
    port=int(os.getenv("PORT", "8099")),
    title="生命食谱 | Life Cookbook",
    favicon="🧬",
    reload=False,
)

