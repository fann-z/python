// 生成二维码
document.addEventListener('DOMContentLoaded', function() {
    new QRCode(document.getElementById("qrCode"), {
        text: window.location.href,
        width: 256,
        height: 256,
        colorDark: "#1a237e",
        colorLight: "#ffffff",
        correctLevel: QRCode.CorrectLevel.H
    });
});

// 文件上传处理
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.onclick = () => fileInput.click();

dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#00bcd4';
};

dropZone.ondragleave = (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '';
};

dropZone.ondrop = (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    handleFiles(files);
};

fileInput.onchange = () => {
    handleFiles(fileInput.files);
};

function handleFiles(files) {
    [...files].forEach(file => {
        uploadFile(file);
    });
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        xhr: function() {
            const xhr = new XMLHttpRequest();
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = (e.loaded / e.total) * 100;
                    $('.progress-bar').css('width', percent + '%');
                }
            });
            return xhr;
        },
        beforeSend: function() {
            $('.progress').show();
        },
        success: function(response) {
            showNotification('上传成功', 'success');
            refreshFileList();
        },
        error: function() {
            showNotification('上传失败', 'error');
        },
        complete: function() {
            setTimeout(() => {
                $('.progress').hide();
                $('.progress-bar').css('width', '0%');
            }, 1000);
        }
    });
}

function refreshFileList() {
    $.get('/files', function(files) {
        const fileList = document.getElementById('fileList');
        fileList.innerHTML = '';
        
        files.forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <i class="material-icons file-icon">description</i>
                <div class="flex-grow-1">${file.name}</div>
                <div class="action-buttons">
                    <button class="custom-button" onclick="downloadFile('${file.name}')">
                        <i class="material-icons">download</i>
                    </button>
                    <button class="custom-button" onclick="deleteFile('${file.name}')">
                        <i class="material-icons">delete</i>
                    </button>
                </div>
            `;
            fileList.appendChild(fileItem);
        });
    });
}

function downloadFile(filename) {
    window.location.href = `/download/${filename}`;
}

function deleteFile(filename) {
    if (confirm('确定要删除这个文件吗？')) {
        $.ajax({
            url: `/delete/${filename}`,
            type: 'DELETE',
            success: function() {
                showNotification('删除成功', 'success');
                refreshFileList();
            },
            error: function() {
                showNotification('删除失败', 'error');
            }
        });
    }
}

function showNotification(message, type) {
    const notification = $(`
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    $('body').append(notification);
    setTimeout(() => notification.remove(), 3000);
}

// 初始加载文件列表
refreshFileList();