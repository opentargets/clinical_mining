from sqlmodel import create_engine, select, Session, SQLModel
from pyspark.sql import SparkSession


def connect_to_db(DATABASE_URL):
    engine = create_engine(DATABASE_URL)
    SQLModel.metadata.create_all(engine)
    return engine


def select_table(entity, engine):
    with Session(engine) as session:
        statement = select(entity)
        records = session.exec(statement).all()
        return (
            SparkSession.builder.getOrCreate()
            .createDataFrame([e.model_dump() for e in records])
            .distinct()
        )
