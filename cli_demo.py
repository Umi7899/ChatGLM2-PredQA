import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
import lottery_predicter_pl_end

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = """现在，请你扮演彩票预测模型。已知根据AI模型在历史开奖数据上的分析，预测得到本周可能的开奖结果为{pred}。

请你根据开奖结果回答用户的问题，用户的问题是{query}"""
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    prediction = lottery_predicter_pl_end.unique_cc_str
    past_key_values, history = None, []
    global stop_stream
    print("欢迎使用预测模型！输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    # 若为第一轮对话，则加入引导prompt
    cnt = 0 #记录轮次
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print("欢迎使用预测模型！输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            cnt = 0
            continue
        new_query = """现在，请你扮演彩票预测模型。已知根据AI模型在历史开奖数据上的分析，预测得到本周可能的开奖结果为{pred}。
        请你根据开奖结果回答用户的问题，用户的问题是{question}"""
        new_query = new_query.replace("{question}", query).replace("{pred}", prediction)
        print("\nChatGLM：", end="")
        current_length = 0

        if cnt == 0:
            for response, history, past_key_values in model.stream_chat(tokenizer, new_query, history=history,
                                                                        past_key_values=past_key_values,
                                                                        return_past_key_values=True):
                if stop_stream:
                    stop_stream = False
                    break
                else:
                    print(response[current_length:], end="", flush=True)
                    current_length = len(response)
        else:
            for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                        past_key_values=past_key_values,
                                                                        return_past_key_values=True):
                if stop_stream:
                    stop_stream = False
                    break
                else:
                    print(response[current_length:], end="", flush=True)
                    current_length = len(response)
        print("")

        cnt = cnt + 1


if __name__ == "__main__":
    main()
