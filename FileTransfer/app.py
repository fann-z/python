from flask import Flask, request, render_template, send_from_directory, jsonify, make_response
import os
import qrcode
from werkzeug.utils import secure_filename
import socket
from datetime import datetime
import humanize
import logging
from logging.handlers import RotatingFileHandler
import shutil

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
QR_CODE_PATH = os.path.join(STATIC_FOLDER, 'qr_code.png')
LOG_FOLDER = 'logs'
ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx',
    'xls', 'xlsx', 'zip', 'rar', '7z', 'mp3', 'mp4', 'avi'
}
MAX_FILE_SIZE = 512 * 1024 * 1024  # 512MB

# 应用配置
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=MAX_FILE_SIZE,
    JSON_AS_ASCII=False
)

# 确保必要的文件夹存在
for folder in [UPLOAD_FOLDER, STATIC_FOLDER, LOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 配置日志
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler(
    os.path.join(LOG_FOLDER, 'app.log'),
    maxBytes=10000000,
    backupCount=5
)
handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
app.logger.addHandler(handler)


def allowed_file(filename):
    """检查文件类型是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_size_format(size):
    """返回人类可读的文件大小格式"""
    return humanize.naturalsize(size)


def get_local_ip():
    """获取本地IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        app.logger.error(f"获取本地IP失败: {str(e)}")
        return '127.0.0.1'


def generate_qr_code(url):
    """生成并保存二维码"""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img.save(QR_CODE_PATH)
        return True
    except Exception as e:
        app.logger.error(f"生成二维码失败: {str(e)}")
        return False


def get_file_info(filepath):
    """获取文件信息"""
    try:
        stats = os.stat(filepath)
        return {
            'name': os.path.basename(filepath),
            'size': get_file_size_format(stats.st_size),
            'size_bytes': stats.st_size,
            'modified': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'created': datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'extension': os.path.splitext(filepath)[1][1:].lower()
        }
    except Exception as e:
        app.logger.error(f"获取文件信息失败: {str(e)}")
        return None


@app.route('/')
def index():
    """主页路由"""
    local_ip = get_local_ip()
    qr_url = f'http://{local_ip}:5000'

    # 生成二维码
    generate_qr_code(qr_url)

    # 获取系统信息
    system_info = {
        'total_space': get_file_size_format(shutil.disk_usage(UPLOAD_FOLDER).total),
        'used_space': get_file_size_format(shutil.disk_usage(UPLOAD_FOLDER).used),
        'free_space': get_file_size_format(shutil.disk_usage(UPLOAD_FOLDER).free)
    }

    return render_template('index.html',
                           qr_url=qr_url,
                           system_info=system_info,
                           max_file_size=get_file_size_format(MAX_FILE_SIZE))


@app.route('/upload', methods=['POST'])
def upload_file():
    """文件上传处理"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 检查是否存在同名文件
        if os.path.exists(filepath):
            base, extension = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                filename = f"{base}_{counter}{extension}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                counter += 1

        file.save(filepath)
        file_info = get_file_info(filepath)

        app.logger.info(f"文件上传成功: {filename}")
        return jsonify({
            'message': '文件上传成功',
            'file': file_info
        })

    except Exception as e:
        app.logger.error(f"文件上传失败: {str(e)}")
        return jsonify({'error': '文件上传失败'}), 500


@app.route('/files')
def list_files():
    """获取文件列表"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                file_info = get_file_info(filepath)
                if file_info:
                    files.append(file_info)

        # 按修改时间排序
        files.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify(files)

    except Exception as e:
        app.logger.error(f"获取文件列表失败: {str(e)}")
        return jsonify({'error': '获取文件列表失败'}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """文件下载处理"""
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            as_attachment=True
        )
    except Exception as e:
        app.logger.error(f"文件下载失败: {str(e)}")
        return jsonify({'error': '文件下载失败'}), 404


@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """文件删除处理"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404

        os.remove(filepath)
        app.logger.info(f"文件删除成功: {filename}")
        return jsonify({'message': '文件删除成功'})

    except Exception as e:
        app.logger.error(f"文件删除失败: {str(e)}")
        return jsonify({'error': '文件删除失败'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """处理文件过大的错误"""
    return jsonify({
        'error': f'文件大小超过限制 (最大 {get_file_size_format(MAX_FILE_SIZE)})'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """处理内部服务器错误"""
    app.logger.error(f"服务器错误: {str(error)}")
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)