import asyncio
import aiohttp
import time
import random
from collections import defaultdict

random.seed(42)

URL = ""
HEADERS = {"Content-Type": "application/json"}

USER_COUNT = 5
SIMULATION_DURATION = 180
POST_DATA_TEMPLATE = {
    "model": "phi-4",
    "messages": [
        {"role": "user", "content": "What is the capital of Japan?"}
    ]
}

user_start_times = {}
user_response_times = defaultdict(list)

experiment_start_time = None

def get_start_delay():
    delay = random.normalvariate(mu=60, sigma=40)
    return max(0, min(SIMULATION_DURATION, delay))

async def simulated_user(user_id: int, session: aiohttp.ClientSession):
    global experiment_start_time

    start_delay = get_start_delay()
    await asyncio.sleep(start_delay)

    user_start_times[user_id] = time.perf_counter() - experiment_start_time

    for i in range(1, 6):
        post_data = POST_DATA_TEMPLATE.copy()
        start_time = time.perf_counter()
        try:
            async with session.post(URL, headers=HEADERS, json=post_data) as response:
                resp_json = await response.json()
                elapsed = time.perf_counter() - start_time
                user_response_times[user_id].append(elapsed)
                print(f"[User {user_id} | Request {i}] Delay {start_delay:.1f}s | Status: {response.status}, Time: {elapsed:.3f}s")
                #print("    ", resp_json["choices"][0]["message"]["content"])
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            user_response_times[user_id].append(elapsed)
            print(f"[User {user_id} | Request {i}] Error: {e}, Time: {elapsed:.3f}s")
        
        if i < 5:
            wait = random.uniform(5, 10)
            await asyncio.sleep(wait)

async def main():
    global experiment_start_time
    experiment_start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [simulated_user(user_id=i+1, session=session) for i in range(USER_COUNT)]
        await asyncio.gather(*tasks)

    print("\n=== ユーザー別集計結果 ===")
    print(f"{'User':<8} {'Start(s)':>10} {'Avg Resp Time(s)':>20} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for user_id in sorted(user_start_times.keys()):
        times = user_response_times[user_id]
        avg_resp = sum(times) / len(times)
        min_resp = min(times)
        max_resp = max(times)
        print(f"{f'User {user_id:02}':<8} {user_start_times[user_id]:>10.1f} {avg_resp:>20.3f} {min_resp:>10.3f} {max_resp:>10.3f}")

if __name__ == "__main__":
    asyncio.run(main())
