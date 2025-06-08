import requests
import random
import time

def get_random_user_agent():
    """
    Return a random, modern user agent string from a list of browsers.
    """
    user_agents = [
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:110.0) "
         "Gecko/20100101 Firefox/110.0"),
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1_0) "
         "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15"),
    ]
    return random.choice(user_agents)

# def get_random_proxy():
#     """
#     Return a random proxy from a pre-defined list.
#     Proxies can be in the form:
#       - http://username:password@ip:port
#       - https://ip:port
#       - socks5://ip:port
#     Make sure the proxies are valid and actively working.
#     """
#     proxies = [
#         "http://123.45.67.89:8080",
#         "http://98.76.54.32:8080",
#         # Add as many working proxies as you can for better rotation
#     ]
#     return random.choice(proxies)

def rotate_session(session):
    """
    Rotates the session's User-Agent and proxies to impersonate 
    a different 'browser' and IP address each time.
    """
    session.headers.update({"User-Agent": get_random_user_agent()})
    
    # If you want to rotate other headers (like Sec-Ch-Ua, etc.), do it here as well.
    # session.headers.update({"Sec-Ch-Ua": '...'})
    
    # proxy = get_random_proxy()
    # session.proxies = {
    #     "http": proxy,
    #     "https": proxy
    # }
    return session

def human_delay(min_sec = 1, max_sec = 59):
    """
    Sleep for a random duration between min_sec and max_sec, 
    imitating human-like pauses between actions.
    Set to 2 request per min.
    """
    time.sleep(random.uniform(min_sec, max_sec))

def main():
    # Constants / Configuration
    BRAND = "musinsastandard"
    MAX_ITEMS = 10
    
    BRAND_PRODUCTS_URL = "https://api.musinsa.com/api2/dp/v1/plp/goods"
    PRODUCT_REVIEW_URL = "https://goods.musinsa.com/api2/review/v1/view/list"
    
    params_brand = {
        "gf": "A",
        "sortCode": "POPULAR",
        "brand": BRAND,
        "page": 1,
        "size": 30,
        "caller": "FLAGSHIP"
    }

    # Create a requests session
    session = requests.Session()
    
    # Set base headers often sent by a real browser (beyond User-Agent).
    # We will rotate the User-Agent via rotate_session.
    session.headers.update({
        "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                   "image/avif,image/webp,image/apng,*/*;q=0.8,"
                   "application/signed-exchange;v=b3;q=0.9"),
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": f"https://www.musinsa.com/brand/{BRAND}?gf=A",
        "Sec-Ch-Ua": '"Chromium";v="111", "Not A(Brand";v="24", "Google Chrome";v="111"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    })

    # Rotate session to initialize random proxy + user agent.
    session = rotate_session(session)

    # Make a request for the brand products
    # If the current proxy fails, you might consider catching exceptions and trying a new proxy.
    response = session.get(BRAND_PRODUCTS_URL, params=params_brand, timeout=10)
    response.raise_for_status()
    products_json = response.json()
    
    # Extract product list
    product_list = products_json.get("data", {}).get("list", [])
    
    items_processed = 0
    
    for product in product_list:
        if items_processed >= MAX_ITEMS:
            break  # Stop if we've reached the maximum we want

        goods_no = product.get('goodsNo')
        if not goods_no:
            continue
        goods_no = int(goods_no)
        
        # Prepare review params
        params_review = {
            "page": 0,
            "pageSize": 10,
            "goodsNo": goods_no,
            "sort": "goods_est_asc",  # low-to-high rating
            "selectedSimilarNo": goods_no,
            "myFilter": "false",
            "hasPhoto": "false",
            "isExperience": "false"
        }

        # Random delay to mimic human browsing
        human_delay()

        # Optionally rotate IP/user-agent for each request
        session = rotate_session(session)
        
        review_response = session.get(PRODUCT_REVIEW_URL, params=params_review, timeout=10)
        review_response.raise_for_status()
        reviews_json = review_response.json()
        
        logging.info(f"##### Goods Number: {goods_no} #####")
        logging.info(f"Goods info: {product}\n")
        items_processed += 1

        reviews_list = reviews_json.get("data", {}).get("list", [])
        
        for review in reviews_list:
            grade = review.get('grade', '')
            if not grade:
                continue
            rating = int(grade)
            if 1 <= rating <= 3:
                # create_date = review.get('createDate', 'N/A')
                # review_no = review.get('no', 'N/A')
                # user_info = review.get('userProfileInfo', {})
                # user_nickname = user_info.get('userNickName', 'N/A')
                # content = review.get('content', 'N/A')
                
                logging.info(
                    # f"    Goods Number: {goods_no}\n"
                    # f"    Review Number: {review_no}\n"
                    # f"    Create Date: {create_date}\n"
                    # f"    Rating: {rating}\n"
                    # f"    User Nickname: {user_nickname}\n"
                    # f"    Content: {content}\n"
                    logging.info(f"{review}\n")
                )

if __name__ == "__main__":
    main()
