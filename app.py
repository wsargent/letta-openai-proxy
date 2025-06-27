import os
import time
import uuid
from typing import Generator, Union

import uvicorn
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from hayhooks import BasePipelineWrapper, create_app
from hayhooks.server.pipelines import registry
from hayhooks.server.routers import openai as openai_module_to_patch
from hayhooks.server.routers.openai import ChatCompletion, ChatRequest, Choice, Message, ModelObject, ModelsResponse
from hayhooks.settings import settings
from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer
from letta_client import Letta
from loguru import logger as log


HAYSTACK_DETAILED_TRACING = False

if HAYSTACK_DETAILED_TRACING:
    # https://docs.haystack.deepset.ai/docs/logging
    tracing.tracer.is_content_tracing_enabled = True  # to enable tracing/logging content (inputs/outputs)
    tracing.enable_tracing(
        LoggingTracer(
            tags_color_strings={
                "haystack.component.input": "\x1b[1;31m",
                "haystack.component.name": "\x1b[1;34m",
            }
        )
    )

# Define the Letta server URL and token
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://letta:8283")
LETTA_API_TOKEN = os.getenv("LETTA_API_TOKEN", "")


def fetch_letta_models():
    """Fetch available models from Letta server using the Letta client directly"""
    try:
        effective_token = LETTA_API_TOKEN if LETTA_API_TOKEN else None
        # Initialize the Letta client
        client = Letta(base_url=LETTA_BASE_URL, token=effective_token)

        # Get the list of agents
        agents = client.agents.list()

        # Filter out agents with names ending in "sleeptime"
        return [{"id": agent.id, "name": agent.name} for agent in agents if not agent.name.endswith("sleeptime")]
    except Exception as e:
        log.error(f"Unexpected error when fetching agents from Letta: {e}", exc_info=True)
        return []


async def get_models_override():
    """
    Override of the OpenAI /models endpoint to return Letta models.

    This returns a list of available Letta agents as OpenAI-compatible models.
    """
    letta_models = fetch_letta_models()

    return ModelsResponse(
        data=[
            ModelObject(
                id=model["id"],
                name=model["name"],
                object="model",
                created=int(time.time()),
                owned_by="letta",
            )
            for model in letta_models
        ],
        object="list",
    )


openai_module_to_patch.get_models = get_models_override

for route_idx, route in enumerate(openai_module_to_patch.router.routes):
    if isinstance(route, APIRoute) and route.path in ["/models", "/v1/models"]:
        route.endpoint = get_models_override


async def chat_completions_override(chat_req: ChatRequest) -> Union[ChatCompletion, StreamingResponse]:
    # Get the letta_proxy pipeline wrapper
    # Assuming 'letta_proxy' is the registered name of your pipeline
    pipeline_wrapper = registry.get("letta_proxy")

    if not pipeline_wrapper:
        log.error("Pipeline 'letta_proxy' not found in registry.")
        raise HTTPException(status_code=500, detail="Chat backend pipeline 'letta_proxy' not found.")

    if not isinstance(pipeline_wrapper, BasePipelineWrapper):
        log.error(f"Retrieved 'letta_proxy' is not a BasePipelineWrapper instance. Type: {type(pipeline_wrapper)}")
        raise HTTPException(status_code=500, detail="Chat backend pipeline 'letta_proxy' is of an unexpected type.")

    if not pipeline_wrapper._is_run_chat_completion_implemented:  # Now Pylance should be happier after isinstance
        log.error(f"Pipeline 'letta_proxy' (type: {type(pipeline_wrapper)}) does not implement run_chat_completion.")
        raise HTTPException(status_code=501, detail="Chat completions endpoint not implemented for 'letta_proxy' model.")

    request_body_dump = chat_req.model_dump()
    if "agent_id" not in request_body_dump:
        request_body_dump["agent_id"] = chat_req.model
        log.info(f"Injected agent_id='{chat_req.model}' into request_body_dump for letta_proxy.")

    try:
        result_generator = await run_in_threadpool(
            pipeline_wrapper.run_chat_completion,
            model=chat_req.model,
            messages=chat_req.messages,
            body=request_body_dump,
        )
    except ValueError as ve:
        log.error(f"ValueError in letta_proxy.run_chat_completion: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.error(f"Exception calling letta_proxy.run_chat_completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing chat request with letta_proxy.")

    resp_id = f"chatcmpl-{uuid.uuid4()}"  # OpenAI compatible ID

    def stream_chunks() -> Generator[str, None, None]:
        try:
            for chunk_content in result_generator:
                if not isinstance(chunk_content, str):
                    log.warning(f"letta_proxy returned non-string chunk: {type(chunk_content)}. Converting to str.")
                    chunk_content = str(chunk_content)

                chunk_resp = ChatCompletion(
                    id=resp_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=chat_req.model,
                    choices=[Choice(index=0, delta=Message(role="assistant", content=chunk_content))],
                )
                yield f"data: {chunk_resp.model_dump_json()}\n\n"

            final_chunk = ChatCompletion(
                id=resp_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=chat_req.model,
                choices=[Choice(index=0, delta=Message(role="assistant", content=""), finish_reason="stop")],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
        except Exception as e:
            log.error(f"Error during streaming from letta_proxy: {e}", exc_info=True)
            error_chunk_content = f"Error processing stream: {e}"
            error_resp = ChatCompletion(
                id=resp_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=chat_req.model,
                choices=[Choice(index=0, delta=Message(role="assistant", content=error_chunk_content), finish_reason="stop")],
            )
            yield f"data: {error_resp.model_dump_json()}\n\n"

    if chat_req.stream:
        log.info(f"Returning StreamingResponse for model {chat_req.model}")
        return StreamingResponse(stream_chunks(), media_type="text/event-stream")
    else:
        # Non-streaming: collect all chunks and return a single ChatCompletion
        log.info(f"Returning non-streaming ChatCompletion for model {chat_req.model}")
        full_response_content = ""
        try:
            for chunk_content in result_generator:
                if not isinstance(chunk_content, str):
                    log.warning(f"letta_proxy returned non-string chunk (non-streaming): {type(chunk_content)}. Converting to str.")
                    chunk_content = str(chunk_content)
                full_response_content += chunk_content

            final_resp = ChatCompletion(
                id=resp_id,
                object="chat.completion",
                created=int(time.time()),
                model=chat_req.model,
                choices=[Choice(index=0, message=Message(role="assistant", content=full_response_content), finish_reason="stop")],
            )
            return final_resp
        except Exception as e:
            log.error(f"Error during non-streaming from letta_proxy: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error collecting stream from letta_proxy: {e}")


for route_idx, route in enumerate(openai_module_to_patch.router.routes):
    if isinstance(route, APIRoute):
        if route.path in ["/models", "/v1/models"]:
            route.endpoint = get_models_override
        elif route.path in ["/chat/completions", "/v1/chat/completions"] or route.operation_id == "chat_completions":  # covers /{pipeline_name}/chat
            route.endpoint = chat_completions_override

hayhooks = create_app()

if __name__ == "__main__":
    # Run the combined Hayhooks + MCP server
    uvicorn.run("app:hayhooks", host=settings.host, port=settings.port)