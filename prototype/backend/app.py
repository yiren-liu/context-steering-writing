import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cos.core import multi_contextual_steering_hf
from cos.utils import load_hf_model_and_tokenizer

load_dotenv()


@dataclass
class AppState:
    model: Any
    tokenizer: Any
    is_chat: bool


app_state: Optional[AppState] = None


class GenerateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    style_a: str = Field(..., min_length=1)
    style_b: str = Field(..., min_length=1)
    lambda_a: Optional[float] = Field(None, ge=0.0)
    lambda_b: Optional[float] = Field(None, ge=0.0)
    max_gen_len: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.6, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    draft_text: str
    used_lambda_a: float
    used_lambda_b: float


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    style_a: str = Field(..., min_length=1)
    style_b: str = Field(..., min_length=1)
    edited_text: str = Field(..., min_length=1)


class FeedbackResponse(BaseModel):
    inferred_lambda_a: float
    inferred_lambda_b: float


class ProfileResponse(BaseModel):
    user_id: str
    lambda_a: float
    lambda_b: float
    updated_at: str


def _get_profile_store_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "profiles.json")


def _read_profiles() -> Dict[str, Dict[str, Any]]:
    path = _get_profile_store_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_profiles(profiles: Dict[str, Dict[str, Any]]) -> None:
    path = _get_profile_store_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=True, indent=2)


def _upsert_profile(user_id: str, lambda_a: float, lambda_b: float) -> ProfileResponse:
    profiles = _read_profiles()
    now = datetime.now(timezone.utc).isoformat()
    profiles[user_id] = {
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "updated_at": now,
    }
    _write_profiles(profiles)
    return ProfileResponse(user_id=user_id, lambda_a=lambda_a, lambda_b=lambda_b, updated_at=now)


def _get_profile(user_id: str) -> Optional[ProfileResponse]:
    profiles = _read_profiles()
    entry = profiles.get(user_id)
    if not entry:
        return None
    return ProfileResponse(
        user_id=user_id,
        lambda_a=float(entry.get("lambda_a", 0.0)),
        lambda_b=float(entry.get("lambda_b", 0.0)),
        updated_at=str(entry.get("updated_at", "")),
    )


def _require_state() -> AppState:
    if app_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return app_state


app = FastAPI(title="CoS Writing POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global app_state
    model_name = os.getenv("COS_MODEL_NAME", "llama-2-7b-chat")
    is_chat = os.getenv("COS_IS_CHAT", "false").lower() == "true"
    model, tokenizer = load_hf_model_and_tokenizer(model_name=model_name)
    app_state = AppState(model=model, tokenizer=tokenizer, is_chat=is_chat)


@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest) -> GenerateResponse:
    state = _require_state()
    profile = _get_profile(req.user_id)
    lambda_a = req.lambda_a if req.lambda_a is not None else (profile.lambda_a if profile else 1.0)
    lambda_b = req.lambda_b if req.lambda_b is not None else (profile.lambda_b if profile else 1.0)
    outputs = multi_contextual_steering_hf(
        state.model,
        state.tokenizer,
        prompts=[req.prompt],
        all_contexts=[[req.style_a], [req.style_b]],
        all_lambdas=[[lambda_a], [lambda_b]],
        put_context_first=True,
        is_chat=state.is_chat,
        max_gen_len=req.max_gen_len,
        temperature=req.temperature,
        top_p=req.top_p,
        show_progress=False,
    )
    draft = outputs["generation"][0]
    if isinstance(draft, dict):
        draft_text = draft.get("content", "")
    else:
        draft_text = str(draft)
    return GenerateResponse(
        draft_text=draft_text,
        used_lambda_a=lambda_a,
        used_lambda_b=lambda_b,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    state = _require_state()
    from cos.core import get_multi_cos_logprob_hf

    lambdas = [round(x * 0.25, 2) for x in range(0, 13)]
    scores = get_multi_cos_logprob_hf(
        model=state.model,
        tokenizer=state.tokenizer,
        prompts=[req.prompt],
        all_contexts=[[req.style_a], [req.style_b]],
        responses=[req.edited_text],
        all_lambdas=[lambdas, lambdas],
        is_chat=state.is_chat,
        put_context_first=True,
        temperature=0.6,
        show_progress=False,
    )
    total_logprobs = scores["total_logprobs"]
    best_idx = int(total_logprobs.argmax().item())
    lambda_a = scores["lambdas_a"][best_idx]
    lambda_b = scores["lambdas_b"][best_idx]
    _upsert_profile(req.user_id, lambda_a, lambda_b)
    return FeedbackResponse(inferred_lambda_a=lambda_a, inferred_lambda_b=lambda_b)


@app.get("/profile/{user_id}", response_model=ProfileResponse)
def get_profile(user_id: str) -> ProfileResponse:
    profile = _get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return profile
