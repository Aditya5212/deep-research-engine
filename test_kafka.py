import asyncio
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

BOOTSTRAP = "localhost:9092"
TOPIC = "agent-tasks"


async def test():
    # --- Producer ---
    producer = AIOKafkaProducer(bootstrap_servers=BOOTSTRAP)
    await producer.start()
    try:
        await producer.send_and_wait(TOPIC, b"hello from aiokafka!")
        print("Produced: hello from aiokafka!")
    finally:
        await producer.stop()

    # --- Consumer ---
    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset="earliest",
        group_id="test-group",
    )
    await consumer.start()
    try:
        msg = await asyncio.wait_for(consumer.getone(), timeout=5.0)
        print(f"Consumed: {msg.value.decode()}")
    finally:
        await consumer.stop()


asyncio.run(test())
