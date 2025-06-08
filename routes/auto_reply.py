from flask import Blueprint, request
from review_utils import generate_response_with_openai

auto_reply_bp = Blueprint('auto_reply', __name__)

@auto_reply_bp.route('/generate_auto_reply', methods=['POST'])
def generate_auto_reply():
    try:
        review_content = request.form.get('review_content') or (request.json and request.json.get('review_content'))
        if not review_content:
            return '리뷰 내용이 없습니다.', 400
        reply = generate_response_with_openai(review_content)
        return reply, 200
    except Exception as e:
        return f'오류: {str(e)}', 500
