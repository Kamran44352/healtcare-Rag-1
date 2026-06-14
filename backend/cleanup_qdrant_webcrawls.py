"""
Run once to delete all web-crawl vectors from Qdrant.
Usage:  python cleanup_qdrant_webcrawls.py
"""
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

from app.config import settings

DOCUMENT_IDS = [
    "f268e173-f534-4d46-b568-5c76a66b2504",
    "8fb9b60c-96b9-4859-89c8-b674dfd29a15",
    "ec618d32-44da-4dc9-b0e7-ecdf8e21d5a0",
    "505686b9-ea62-406a-9393-27dea13a5896",
    "3e8aa256-cb2c-41d0-bf75-6f06b45d545f",
    "a8cfb010-025c-446f-9b69-53ed759140fc",
    "4832662a-f9ca-4da7-9187-12bdcda62b97",
    "0a701c53-1a46-4a4e-8762-9304f3bc4091",
    "a5bf58db-91ed-40fb-a3b3-b5a8d04203f4",
    "53501e2a-577f-454f-af26-206b8fbccd2e",
    "ee86ce0e-d8f2-416b-b723-324e35656e80",
    "37c04436-2ca8-48d6-9e16-ef493a22ce5d",
    "b4da5a0f-76f9-4985-b9e4-0db283b57f0d",
    "15cc01a6-147e-4f84-97aa-a9f4c64744d8",
    "77bb23d4-eba0-40d8-8a39-7276b5348f75",
    "083cba8d-9f59-40fd-a7d9-a1e8552b8fe8",
    "7c84595c-f5fb-48c5-90ef-f896a961378f",
    "5a6682f5-2833-4360-bddf-374d8df79a23",
    "bf54449d-9717-40ef-ae93-dfff7648dc7e",
    "42f718d4-8229-4522-814e-6ec2aaf0a5d5",
    "cfde63bf-2334-41bf-b953-0d918552fa55",
]


async def main() -> None:
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )

    result = await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=DOCUMENT_IDS),
                )
            ]
        ),
        wait=True,
    )
    print(f"Qdrant delete result: {result}")

    info = await client.get_collection(settings.qdrant_collection)
    count = getattr(info, "vectors_count", None) or getattr(info.points_count, "__int__", lambda: info.points_count)()
    print(f"Collection points remaining: {info.points_count}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
