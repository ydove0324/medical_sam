import gradio as gr
import openai
max_num_layers = 10
max_num_classes = 10

openai.api_key = "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"  # Replace with your key

def predict(message, history):
    history_openai_format = []
    pre_template = {"role":"user","content":
    "请先判断输入的内容与医学是否相关, 如果与医学相关,请输出问题的回答,否则,请输出 \"对不起,这个问题与医学无关,我们无法回答\""}
    history_openai_format.append(pre_template)
    _temp = history.split('\n')
    _temp.pop()
    history_list = []
    for i in range(len(_temp) // 2):
        history_list.append([_temp[i].split(':')[-1],_temp[i+1].split(':')[-1]])
    for human, assistant in history_list:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages= history_openai_format,
        temperature=1.0,
        stream=True
    )

    partial_message = ""
    for chunk in response:
        if len(chunk['choices'][0]['delta']) != 0:
            partial_message = partial_message + chunk['choices'][0]['delta']['content']
            # yield partial_message
    history_list.append([message,partial_message])
    # history_list.append(partial_message)
    history = ""
    for human,assistant in history_list:
        history = history + "user:" + human + '\n'
        history = history + "云杏AI:" + assistant + '\n'
    return history


def slider_update(num_layers, num_classes, color):
    num_color = 1 if color == '单色' else 3
    state = [gr.update(visible=False)] * 8
    for i in range(num_layers - 2):
        state[i] = gr.update(visible=True)
    return state + [gr.update(visible=True)]

with gr.Blocks() as demo:
    with gr.Tab("模型配置"):
        with gr.Tab("FCN模型配置"):
            with gr.Row():
                with gr.Column():
                    with gr.Tab("创建FCN模型"):
                        FCN_num_layers = gr.Slider(3, max_num_layers, 4, label="模型层数", step=1)
                        FCN_num_classes = gr.Slider(1, max_num_classes, 1, label="分类数", step=1)
                        FCN_color = gr.Radio(["灰色", "彩色"], label="图片类型")
                        btn_FCN_sure_0 = gr.Button(value="确定")
                        FCN_slider = []
                        for i in range(max_num_layers-2):
                            FCN_slider.append(gr.Slider(1, 128, 4, label="中间层{}通道数".format(str(i)), step=1, visible=False))
                        btn_FCN_sure_1 = gr.Button(value="确定", visible=False)
                        btn_FCN_sure_0.click(slider_update, inputs=[FCN_num_layers, FCN_num_classes, FCN_color],
                                            outputs=FCN_slider + [btn_FCN_sure_1])
                    with gr.Tab("选择FCN模型"):
                        gr.File()
                with gr.Column():
                    gr.File(show_label='请选择您的数据集文件并配置')
                    button_FCN_train = gr.Button(value="配置")

        with gr.Tab("Unet模型配置"):
            with gr.Row():
                with gr.Column():
                    with gr.Tab("创建Unet模型"):
                        Unet_num_layers = gr.Slider(3, max_num_layers, 4, label="模型层数", step=1)
                        Unet_num_classes = gr.Slider(1, max_num_classes, 1, label="分类数", step=1)
                        Unet_color = gr.Radio(["灰色", "彩色"], label="图片类型")
                        button_Unet_sure_0 = gr.Button(value="确定")
                        Unet_slider = []
                        for i in range(max_num_layers-2):
                            Unet_slider.append(gr.Slider(1, 128, 4, label="中间层{}通道数".format(str(i)), step=1, visible=False))
                        button_Unet_sure_1 = gr.Button(value="确定", visible=False)
                        button_Unet_sure_0.click(slider_update, inputs=[Unet_num_layers, Unet_num_classes, Unet_color],
                                                 outputs=Unet_slider+[button_Unet_sure_1])
                    with gr.Tab("选择Unet模型"):
                        gr.File()
                with gr.Column():
                    gr.File(show_label='请选择您的数据集文件并配置')
                    button_Unet_train = gr.Button(value="配置")

    with gr.Tab("学习图像分割"):
        with gr.Row():
            gr.File(show_label='请请选择您要使用的模型')
        with gr.Row():
            pass
    with gr.Tab("chatbot"):
        user_message = gr.Textbox(label="输入您的消息")
        chat_history = gr.Textbox(label="聊天历史", lines=6)
        chat_button = gr.Button(label="发送")
        
        # 添加按钮点击后的逻辑
        # def update_chat():
            # 此处调用predict函数，并更新chat_history
            # ...
        
        chat_button.click(predict, inputs=[user_message, chat_history], outputs=[chat_history])

demo.launch()
