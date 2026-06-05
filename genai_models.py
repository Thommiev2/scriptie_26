import os

import openai.lib.streaming.chat
from openai import OpenAI
from google import genai
import json
import csv
from pathlib import Path
from utility_functions import normalize
import time

summarization_prompt = 'Vat het volgende stuk teks samen'
summarization_instructions = None
temperature = 0.3


class GenAI:
    def __init__(self, instructions, prompt, model, client):
        self.instructions: str = instructions
        self.prompt: str = prompt
        self.model: str = model
        self.client = client

    def summarize_single_call(self, text, name, category, model):
        pass

    def summarize_small_batch(self):
        pass

    def summarize_large_batch(self):
        pass

    def instruction_format(self, instruction_id, text):
        pass


class GPT(GenAI):
    def __init__(self):
        super().__init__(instructions=summarization_instructions,
                         prompt=summarization_prompt,
                         model='gpt-4o',
                         client=OpenAI())

    def summarize_single_call(self, text, name, category, model):

        timer = time.perf_counter()
        n_tries = 1
        cooldown = 5

        messages = [{"role": 'user',
                     'content': f"{self.prompt}\n{text}"}]
        if self.instructions:
            messages.insert(0, {"role": "system",
                                "content": self.instructions})
        print(f"> Attempting to summarize {name} from dataset {category}")
        while True:

            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3
                )
                break
            except:
                print(f'| Failed to summaries on the {n_tries} time')
                time.sleep(cooldown * n_tries)
                cooldown *= 2

        print(f"< Transcription completed in {round((time.perf_counter() - timer) / 60, 2)} minutes")

        return completion.choices[0].message.content

    def summarize_large_batch(self):
        pass

    def instruction_format(self, instruction_id, text):

        messages = [{"role": 'user',
                     'content': f"{self.prompt}\n{text}"}]
        if self.instructions:
            messages.insert(0, {"role": "system",
                                "content": self.instructions})
        instruction = {
            "custom_id": f"request-{instruction_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            },
        }

        return instruction


class Gemini(GenAI):
    def __init__(self):
        super().__init__(instructions=summarization_instructions,
                         prompt=summarization_prompt,
                         model="gemini-2.5-flash",
                         client=genai.Client())

    def summarize_single_call(self, text, name, category, model):

        timer = time.perf_counter()
        n_tries = 1
        cooldown = 5

        print(f"> Attempting to summarize{' the ground truth of' if model == 'gt' else ''} {name} from dataset {category}{f' transcribed by {model}' if model != 'gt' else ''}")
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.prompt + '\n' + text,
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                        system_instruction=self.instructions
                    ),
                )
                break
            except:
                print(f'| Failed to summaries on the {n_tries} time')
                time.sleep(cooldown * n_tries)
                n_tries += 1

        print(f"< Transcription completed in {round((time.perf_counter() - timer) / 60, 2)} minutes")
        text = response.text.replace('\n', "")

        return text

    def summarize_small_batch(self):
        inline_requests = []
        for file in os.listdir(Path('batch requests')):
            if file.split('_')[0] == self.model:
                with open(Path('batch requests') / file, 'r', encoding='utf-8') as f_r:
                    for r in f_r:
                        if not r:
                            continue
                        r = r.strip()
                        request = json.loads(r)
                        inline_request = {
                            "contents": request['request']['contents'],
                            'config': {
                                'system_instruction': request['request']["systemInstruction"],
                                "temperature": request['request']["generationConfig"]["temperature"]
                            }
                        }
                        inline_requests.append(inline_request)
                        # print(inline_request)
                    f_r.close()

        # # print(inline_requests)
        batch_name = 'inline_requests_batch'

        # batch_job = self.client.batches.create(
        #     model=self.model,
        #     src=inline_requests,
        #     config={
        #         'display_name': batch_name
        #     }
        # )
        #
        # batch_id = batch_job.name
        # completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
        # timer = 0
        #
        # print(f'Attempting to process batch {batch_name}, {timer / 60} minutes')
        # batch_job = self.client.batches.get(name=batch_id)
        #
        # while batch_job.state.name not in completed_states:
        #     print(f"Current state: {batch_job.state.name}")
        #     time.sleep(10)
        #     batch_job = self.client.batches.get(name=batch_id)
        #
        # print(f"Job finished with state: {batch_job.state.name}")
        #
        # if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        #     result_file_name = batch_job.dest.file_name
        #     print(f"Results are in file: {result_file_name}")
        #     print("Downloading result file content...")
        #     file_content_bytes = self.client.files.download(file=result_file_name)
        #     file_content = file_content_bytes.decode('utf-8')
        #     # The result file is also a JSONL file. Parse and print each line.
        #     for line in file_content.splitlines():
        #         if line:
        #             parsed_response = json.loads(line)
        #             if 'response' in parsed_response and parsed_response['response']:
        #                 for part in parsed_response['response']['candidates'][0]['content']['parts']:
        #                     if part.get('text'):
        #                         print(part['text'])
        #                     elif part.get('inlineData'):
        #                         print(f"Image mime type: {part['inlineData']['mimeType']}")
        #                         data = base64.b64decode(part['inlineData']['data'])
        #             elif 'error' in parsed_response:
        #                 print(f"Error: {parsed_response['error']}")
        #
        # if batch_job.state.name == 'JOB_STATE_FAILED':
        #     print(f"Error: {batch_job.error}")
        #     return None

    def summarize_large_batch(self):
        pass

    def instruction_format(self, request_id, text):

        request = {
            "key": f"request-{request_id}",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": f"{self.prompt}\n{text}"}]}],
                "generationConfig": {
                    "temperature": temperature
                },
            },
        }

        if self.instructions:
            request["request"]["systemInstruction"] = {"parts": [{"text": self.instructions}]}

        return request


def generate_batch(models: list['GenAI'], save=False, normalize_input=False):
    batched_data = os.listdir(Path('batch requests'))

    for genai_model in models:
        print("Initializing generative AI model")
        genai_model = genai_model()
        print(f"model {genai_model.model} initiated succesfully")

        for file in os.listdir(Path('ASR output')):
            batch_file_path = Path('batch requests') / Path(f'{genai_model.model}_{file}.jsonl')
            if str(batch_file_path) not in batched_data:

                with open(batch_file_path, 'w', newline='', encoding='utf-8') as f_w:
                    with open(Path('ASR output') / file, 'r', encoding='utf-8') as f_r:
                        reader = csv.DictReader(f_r)
                        for i, row in enumerate(reader):
                            request_id = f"{i}_{row['name']}_{row['category']}_{row['model']}"
                            text = normalize(row['transcript']) if normalize_input else row['transcript']
                            request = genai_model.instruction_format(request_id, text)

                            json.dump(request, f_w, ensure_ascii=False)
                            f_w.write('\n')
                        f_r.close()
                print(f"generated batch for ASR output file {file}")
                f_w.close()
            else:
                print(f"ASR ouput file {file} has already been converted to a batch")

        # if not save:
        #     yield batch_file_path
        #     batch_file_path.unlink()


# generate_batch([GPT, Gemini])
# a = GPT()
# b = a.summarize_single_call("Fijn dat u vandaag bent gekomen voor het bespreken van de uitslag van de onderzoeken. Zoals u weet hebben we de afgelopen donderdag het aanvullende onderzoek gedaan. En nu heb ik de uitslag. Ja, goedemiddag dokter. Ik ben wel een beetje bang voor de uitslag. Dat kan ik me goed voorstellen. Ik heb helaas geen goed nieuws voor u. Ik moet u mededelen dat u kanker heeft in de dikke darm. Er is een tumor gezien op de scopie. Wat denkt en voelt u nu? Er gaat wel veel in mij om op dit moment. Ik was er al een beetje op voorbereid, maar ik merk dat het toch wel erg confronterend is wat u mij vertelt. Het is ook heel heftig nieuws om te horen. Betekent dat nou dat ik dood ga, dokter? Bent u daar bang voor? Ja, eigenlijk wel. Ik begrijp dat u zich hierover zorgen maakt, maar zijn er genoeg behandelingen?")
# print(b)
