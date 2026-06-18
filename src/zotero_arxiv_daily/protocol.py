from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json

RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]
        prompt = enc.decode(prompt_tokens)
        
        # Groq 모델 인자 주입 안정화 (temperature 조정)
        generation_kwargs = llm_params.get('generation_kwargs', {}).copy()
        if "llama" in generation_kwargs.get("model", "").lower():
            generation_kwargs["temperature"] = min(generation_kwargs.get("temperature", 1.0), 0.3)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **generation_kwargs
        )
        tldr = response.choices[0].message.content
        return tldr.strip() if tldr else ""
    
    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        if self.full_text is not None:
            # Groq Llama 3.1이 헷갈리지 않게 명확한 JSON 형식을 요구하는 프롬프트로 수정
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a JSON array of strings format (e.g., [\"University A\", \"University B\"]). If there is no affiliation found, return an empty array []:\n\n{self.full_text}"
            
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]
            prompt = enc.decode(prompt_tokens)
            
            # Llama-3.1 모델 타겟팅 시 무조건 올바른 JSON만 반환하도록 형식을 강제(response_format)
            generation_kwargs = llm_params.get('generation_kwargs', {}).copy()
            if "llama" in generation_kwargs.get("model", "").lower():
                generation_kwargs["response_format"] = {"type": "json_object"}
                generation_kwargs["temperature"] = 0.0  # 일관된 JSON 출력을 위해 온도를 낮춤
            
            response = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized information extraction assistant. You must output your answer as a valid JSON object containing an 'affiliations' key with a list of strings value. Example format: {\"affiliations\": [\"Tsinghua University\", \"Peking University\"]}. Do not include any explanation or markdown wrappers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **generation_kwargs
            )
            raw_content = response.choices[0].message.content.strip()

            # 안정적인 JSON 파싱 흐름 구성
            try:
                # 1. 완벽한 JSON 오브젝트로 받아왔을 때의 처리
                data = json.loads(raw_content)
                if isinstance(data, dict) and "affiliations" in data:
                    affiliations_list = data["affiliations"]
                elif isinstance(data, list):
                    affiliations_list = data
                else:
                    affiliations_list = []
            except json.JSONDecodeError:
                # 2. 만약 문자열 찌꺼기가 남아있을 때를 위한 폴백(Fallback) 방어코드
                match = re.search(r'\[.*?\]', raw_content, flags=re.DOTALL)
                if match:
                    affiliations_list = json.loads(match.group(0))
                else:
                    affiliations_list = []

            # 값 정제 및 중복 제거
            if isinstance(affiliations_list, list):
                cleaned_affiliations = list(set([str(a).strip() for a in affiliations_list if a]))
                return cleaned_affiliations
            return []
        return []
    
    def generate_affiliations(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client, llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = []
            return []

@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
