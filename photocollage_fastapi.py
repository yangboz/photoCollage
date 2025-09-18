from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import math
from PIL import Image, ImageDraw
from typing import List, Optional
import uvicorn
from pathlib import Path
import io

app = FastAPI(title="PhotoCollage API", description="AI图像拼图生成服务", version="1.0")

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要的目录
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 挂载静态文件服务
app.mount("/static", StaticFiles(directory=str(RESULTS_DIR)), name="static")

# 模板配置
TEMPLATE_CONFIG = {
    "grid": {"free": True, "name": "网格模板"},
    "free": {"free": True, "name": "自由排布模板"},
    "heart": {"free": False, "name": "心形模板", "price": 9.9}
}

class CollageGenerator:
    """拼图生成器"""
    
    def __init__(self):
        self.output_size = (800, 800)
        self.background_color = (255, 255, 255)
    
    def resize_image(self, image: Image.Image, size: tuple) -> Image.Image:
        """等比例缩放图片"""
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
    
    def create_grid_collage(self, images: List[Image.Image]) -> Image.Image:
        """生成网格拼图"""
        num_images = len(images)
        
        # 计算网格布局
        if num_images <= 4:
            cols = rows = 2
        elif num_images <= 9:
            cols = rows = 3
        else:
            cols = 4
            rows = math.ceil(num_images / cols)
        
        # 创建画布
        canvas = Image.new('RGB', self.output_size, self.background_color)
        
        # 计算每个图片的大小
        img_width = self.output_size[0] // cols
        img_height = self.output_size[1] // rows
        
        for i, img in enumerate(images[:cols*rows]):
            if i >= cols * rows:
                break
                
            # 缩放图片
            resized_img = self.resize_image(img.copy(), (img_width-4, img_height-4))
            
            # 计算位置
            row = i // cols
            col = i % cols
            x = col * img_width + 2
            y = row * img_height + 2
            
            # 粘贴图片
            canvas.paste(resized_img, (x, y))
        
        return canvas
    
    def create_free_collage(self, images: List[Image.Image]) -> Image.Image:
        """生成自由排布拼图"""
        canvas = Image.new('RGB', self.output_size, self.background_color)
        num_images = len(images)
        
        # 自由排布的预设位置和大小
        positions = [
            (50, 50, 300, 250),    # x, y, width, height
            (400, 100, 250, 200),
            (100, 350, 200, 180),
            (350, 400, 280, 220),
            (600, 50, 150, 150),
            (50, 600, 180, 150),
            (650, 300, 120, 120),
            (500, 650, 160, 100),
            (300, 200, 120, 100)
        ]
        
        for i, img in enumerate(images):
            if i >= len(positions):
                break
                
            x, y, w, h = positions[i]
            
            # 确保不超出画布边界
            if x + w > self.output_size[0]:
                w = self.output_size[0] - x - 10
            if y + h > self.output_size[1]:
                h = self.output_size[1] - y - 10
            
            # 缩放图片
            resized_img = self.resize_image(img.copy(), (w, h))
            
            # 粘贴图片
            canvas.paste(resized_img, (x, y))
        
        return canvas
    
    def create_heart_collage(self, images: List[Image.Image]) -> Image.Image:
        """生成心形拼图（收费功能）"""
        canvas = Image.new('RGB', self.output_size, self.background_color)
        
        # 心形的数学参数
        center_x, center_y = self.output_size[0] // 2, self.output_size[1] // 2
        scale = 100
        
        # 计算心形上的点位
        heart_points = []
        for i in range(len(images)):
            t = 2 * math.pi * i / len(images)
            # 心形参数方程
            x = scale * (16 * math.sin(t)**3)
            y = -scale * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            heart_points.append((int(center_x + x), int(center_y + y)))
        
        # 在心形点位放置图片
        for i, (img, (x, y)) in enumerate(zip(images, heart_points)):
            size = max(80, 150 - i * 5)  # 图片大小递减
            resized_img = self.resize_image(img.copy(), (size, size))
            
            # 计算粘贴位置（居中）
            paste_x = x - size // 2
            paste_y = y - size // 2
            
            # 确保在画布范围内
            paste_x = max(0, min(paste_x, self.output_size[0] - size))
            paste_y = max(0, min(paste_y, self.output_size[1] - size))
            
            canvas.paste(resized_img, (paste_x, paste_y))
        
        return canvas

# 全局拼图生成器实例
collage_gen = CollageGenerator()

def validate_template(template: str, is_paid: bool = False):
    """验证模板是否可用"""
    if template not in TEMPLATE_CONFIG:
        raise HTTPException(status_code=400, detail="不支持的模板类型")
    
    config = TEMPLATE_CONFIG[template]
    if not config["free"] and not is_paid:
        raise HTTPException(
            status_code=402, 
            detail=f"模板 '{config['name']}' 需要付费使用，价格：¥{config['price']}"
        )

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "PhotoCollage API 服务已启动",
        "version": "1.0",
        "templates": TEMPLATE_CONFIG,
        "docs": "/docs"
    }

@app.get("/templates")
async def get_templates():
    """获取所有模板信息"""
    return {"templates": TEMPLATE_CONFIG}

@app.post("/collage")
async def create_collage(
    images: List[UploadFile] = File(..., description="上传的图片文件列表"),
    template: str = Form(..., description="拼图模板: grid/heart/free"),
    paid: bool = Form(False, description="是否已付费（心形模板需要）")
):
    """
    生成拼图
    - **images**: 上传的图片文件列表（最多9张）
    - **template**: 拼图模板 (grid: 网格, heart: 心形, free: 自由排布)
    - **paid**: 是否已付费（心形模板需要付费）
    """
    
    # 验证图片数量
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="至少需要上传一张图片")
    if len(images) > 9:
        raise HTTPException(status_code=400, detail="最多支持上传9张图片")
    
    # 验证模板
    validate_template(template, paid)
    
    try:
        # 处理上传的图片
        processed_images = []
        for upload_file in images:
            # 验证文件类型
            if not upload_file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"文件 {upload_file.filename} 不是有效的图片格式")
            
            # 读取图片
            image_data = await upload_file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            processed_images.append(image)
        
        # 根据模板生成拼图
        if template == "grid":
            result_image = collage_gen.create_grid_collage(processed_images)
        elif template == "free":
            result_image = collage_gen.create_free_collage(processed_images)
        elif template == "heart":
            result_image = collage_gen.create_heart_collage(processed_images)
        else:
            raise HTTPException(status_code=400, detail="不支持的模板类型")
        
        # 保存结果图片
        filename = f"{uuid.uuid4().hex}.jpg"
        result_path = RESULTS_DIR / filename
        result_image.save(result_path, "JPEG", quality=90)
        
        return {
            "success": True,
            "message": "拼图生成成功",
            "template": template,
            "template_name": TEMPLATE_CONFIG[template]["name"],
            "image_count": len(processed_images),
            "result_url": f"/results/{filename}",
            "download_url": f"/static/{filename}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"拼图生成失败: {str(e)}")

@app.get("/results/{filename}")
async def get_result_image(filename: str):
    """
    获取生成的拼图图片
    - **filename**: 图片文件名
    """
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="图片文件不存在")
    
    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=filename
    )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "PhotoCollage API",
        "version": "1.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )