from sqlmodel import SQLModel, Field

class Studies(SQLModel, table=True, extend_existing=True):
    __table_args__ = {'extend_existing': True}
    nct_id: str = Field(default=None, primary_key=True)
    overall_status: str
    phase: str
    study_type: str
    is_fda_regulated_drug: bool # if study evaluates a drug or biological product subject to US FDA regulation

class Interventions(SQLModel, table=True, extend_existing=True):
    __table_args__ = {'extend_existing': True}
    id: int = Field(default=None, primary_key=True)
    nct_id: str
    intervention_type: str
    name: str

class Browse_Interventions(SQLModel, table=True, extend_existing=True):
    __table_args__ = {'extend_existing': True}
    id: int = Field(default=None, primary_key=True)
    nct_id: str
    mesh_term: str
    downcase_mesh_term: str
    mesh_type: str


class Conditions(SQLModel, table=True, extend_existing=True):
    __table_args__ = {'extend_existing': True}
    id: int = Field(default=None, primary_key=True)
    nct_id: str
    downcase_name: str

class Browse_Conditions(SQLModel, table=True, extend_existing=True):
    __table_args__ = {'extend_existing': True}
    id: int = Field(default=None, primary_key=True)
    nct_id: str
    downcase_mesh_term: str
    mesh_type: str
