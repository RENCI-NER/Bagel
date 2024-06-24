from langserve import CustomUserType, add_routes
from config import settings
from chain import ChainFactory
import fastapi

app = fastapi.FastAPI(title="Bagel API server", description="Runs bagel re-ranking prompts against pre-configured LLMs")


class Query(CustomUserType):
    abstract: str
    term: str
    synonym_list: str


for chain in ChainFactory.get_all_chains():
    add_routes(
        app,
        chain,
        path=f"/{chain.name}",
        input_type=Query
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=9001)
