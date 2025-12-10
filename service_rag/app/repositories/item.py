from sqlalchemy.orm import Session
from sqlalchemy import desc
from service_rag.app.schemas.item import KnowledgeItemCreate
from service_rag.app.modules.dataset_module import KnowledgeItemORM

def create_knowledge_item(db:Session, obj: KnowledgeItemCreate):
    try:
        db_item = KnowledgeItemORM(**obj.model_dump())
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
    except Exception as e:
        print(f"{str(e)}")
        raise e
    return db_item

def get_knowledge_item(db:Session):
    return (db.query(KnowledgeItemORM)
            .where(KnowledgeItemORM.deleted != True)
            .all()
           )
def delete_knowledge_item(db:Session, ids):
    db_item = db.query(KnowledgeItemORM).filter(KnowledgeItemORM.id == ids).first()

    if db_item:
        db.delete(db_item)
        db.commit()
    return db_item