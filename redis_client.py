# redis_client.py

import redis


def get_redis():
    # uses your Redis Labs / Redis Cloud instance
    return redis.Redis(
        host="redis-19340.c93.us-east-1-3.ec2.cloud.redislabs.com",
        port=19340,
        username="default",
        password="eAsDFA0HZMTi4y9mHxxwfY6hIcZB2u4Y",
        decode_responses=False,  # we store raw bytes (images)
    )
