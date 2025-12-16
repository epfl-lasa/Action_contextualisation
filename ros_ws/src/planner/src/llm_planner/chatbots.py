from openai import OpenAI
import os
import typing as tp
import pickle
import time


class GptChatBot:

    API_KEY_ENV_VAR = "GPT_API_KEY"
    DFLT_SYSTEM_MSG = "You are a robotics and machine learning expert. You reply using markdown format"

    @property
    def system_msg(self) -> str:
        return self._system_msg

    @system_msg.setter
    def system_msg(self, value: str) -> None:
        self._system_msg = value

    def __init__(self, system_msg: tp.Optional[str] = None,
                 warm_start_file_name: tp.Optional[str] = None,
                 auto_save_file_name: tp.Optional[str] = None) -> None:
        self.client = OpenAI(api_key=os.getenv(self.API_KEY_ENV_VAR))
        self._history = []
        self._system_msg = self.DFLT_SYSTEM_MSG if system_msg is None else system_msg
        self._auto_save_file_name = auto_save_file_name
        self._response_times = []

        if warm_start_file_name is not None:
            self.load_chat(warm_start_file_name)

    def set_history(self, new_history: str):
        self._history = new_history
        # Auto save if needed
        if self._auto_save_file_name is not None:
            self.save_chat(self._auto_save_file_name)
    
    def print_history(self):
        formatted_history = ["GPT:" + self._history[i] if i % 2 else
                             "ASK:" + self._history[i]
                             for i in range(len(self._history))]

        for msg in formatted_history:
            print(msg)

    def ask(self, input: str,
            model: str = "gpt-4-1106-preview",
            show_output: bool = False,
            force_cot: bool = False) -> str:

        # Format messages
        system_msg = [{"role": "system", "content": self._system_msg}]
        self._history.append(input)

        # Forcing chain of thought
        if force_cot:
            self._history.append("Let's thing step by step.")
        user_bot_dialogue = [{"role": "assistant", "content": self._history[i]} if i % 2 else
                             {"role": "user", "content": self._history[i]}
                             for i in range(len(self._history))]
        msgs = system_msg + user_bot_dialogue

        # GPT query... costs money
        tic = time.time()
        response = self.client.chat.completions.create(model=model, messages=msgs)
        toc = time.time()
        self._response_times.append(toc-tic)

        # Check status
        status_code = response.choices[0].finish_reason
        assert status_code == "stop", f"The status code was {status_code}."

        # Save and return output
        text_response = response.choices[0].message.content

        if force_cot:
            text_response = "Let's think step by step.\n " + text_response
        self._history.append(text_response)

        # Show output
        if show_output:
            print(f"GPT: {text_response}")

        # Auto save if needed
        if self._auto_save_file_name is not None:
            self.save_chat(self._auto_save_file_name)

        return text_response

    def interactive_chat(self) -> None:

        # Discussion loop
        print("Ready to chat...")
        while True:

            user_input = input("You: ")
            if user_input in ["quit", "exit", "q"]:
                break

            self.ask(user_input, show_output=True)

    def save_chat(self, file_name: str) -> None:
        with open(file_name, 'wb') as file:
            pickle.dump((self._history, self._response_times), file)

    def load_chat(self, file_name: str) -> None:
        with open(file_name, 'rb') as file:
            pickle_content = pickle.load(file)
            self._history = pickle_content[0]
            self._response_times = pickle_content[1]

    def __call__(self, input: str, model: str = "gpt-3.5-turbo") -> str:

        # Format messages
        system_msg = [{"role": "system", "content": self._system_msg}]
        user_input = [{"role": "assistant", "content": input}]
        msgs = system_msg + user_input

        # GPT query... costs money
        response = self.client.chat.completions.create(model=model, messages=msgs)

        # Check status
        status_code = response.choices[0].finish_reason
        assert status_code == "stop", f"The status code was {status_code}."

        # Save and return output
        text_response = response.choices[0].message.content
        return text_response