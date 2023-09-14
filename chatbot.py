import openai
import gradio as gr

openai.api_key = ""  # Replace with your key

def predict(message, history):
    history_openai_format = []
    pre_template = {"role":"user","content":
    "请先判断输入的内容与医学是否相关, 如果与医学相关,请输出问题的回答,否则,请输出 \"对不起,这个问题与医学无关,我们无法回答\""}
    history_openai_format.append(pre_template)
    for human, assistant in history:
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
            yield partial_message

gr.ChatInterface(predict).queue().launch()