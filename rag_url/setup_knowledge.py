"""웹 URL → 본문 추출 → OpenAI Vector Store 등록 (최초/갱신 시 실행)."""
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json
import re
import os

import requests
from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI

URLS = [
    "https://www.unesco.org/en/articles/ai-competency-framework-teachers",
    "https://www.khanacademy.org/khanmigo-for-educators",
]
OUT = Path("knowledge_web")


def extract_page(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise ValueError(f"올바른 HTTP(S) URL이 아닙니다: {url}")
    response = requests.get(url, timeout=20, headers={"User-Agent": "EducationRAGDemo/1.0"})
    response.raise_for_status()
    if "text/html" not in response.headers.get("Content-Type", "").lower():
        raise ValueError("HTML 페이지가 아닙니다. PDF 등은 별도 처리해야 합니다.")
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    lines = [x.strip() for x in root.get_text("\n", strip=True).splitlines()]
    body = "\n".join(x for x in lines if x)
    if len(body) < 200:
        raise ValueError("추출된 본문이 너무 짧습니다. 동적 페이지 또는 접근 제한 여부를 확인하세요.")
    return f"제목: {title}\n출처 URL: {url}\n\n{body}\n"


def main():
    api_key = os.getenv["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    OUT.mkdir(exist_ok=True)
    paths = []
    for url in URLS:
        try:
            text = extract_page(url)
            name = hashlib.sha256(url.encode()).hexdigest()[:12] + ".txt"
            path = OUT / name
            path.write_text(text, encoding="utf-8")
            paths.append(path)
            print(f"수집 완료: {url} ({len(text)}자)")
        except (requests.RequestException, ValueError) as exc:
            print(f"수집 실패: {url} - {exc}")
    if not paths:
        raise RuntimeError("수집된 웹페이지가 없습니다. URL과 접근 가능 여부를 확인하세요.")

    store = client.vector_stores.create(name="Education Web Knowledge")
    streams = [p.open("rb") for p in paths]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=store.id, files=streams
        )
    finally:
        for stream in streams:
            stream.close()
    print("인덱싱 상태:", batch.status)
    print("파일 상태:", batch.file_counts)
    if batch.status != "completed":
        raise RuntimeError("인덱싱이 완료되지 않았습니다. 파일 상태를 확인하세요.")
    print("\nsecrets.toml에 다음 값을 추가/수정하세요:")
    print(f'VECTOR_STORE_ID = "{store.id}"')
    print("기존 Vector Store는 자동 삭제되지 않습니다. 반복 실행 시 새 저장소와 비용이 발생할 수 있습니다.")


if __name__ == "__main__":
    main()
