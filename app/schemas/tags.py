from pydantic import BaseModel


class TagCreate(BaseModel):
    name: str
    color: str = "#6366f1"


class TagResponse(BaseModel):
    id: int
    name: str
    color: str

    model_config = {"from_attributes": True}


class TagAssignRequest(BaseModel):
    tag_id: int
