"""
文件处理服务 - 集成RAG功能
"""
import os
import uuid
import aiofiles
import logging
from typing import Dict, Any
from pypdf import PdfReader
import docx
from io import BytesIO
import pytesseract
from PIL import Image
import tempfile
import io

from ..config import config
from ..deps import add_documents_to_vector_store

# 配置日志
logger = logging.getLogger(__name__)


class FileService:
    """文件处理服务"""

    def __init__(self):
        # 确保上传目录存在
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    async def save_file(self, file_content: bytes, filename: str) -> str:
        """保存上传的文件"""
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(config.UPLOAD_DIR, unique_filename)

        logger.info(f"💾 保存文件: {unique_filename} -> {file_path}")

        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)

        logger.info(f"✅ 文件保存成功: {file_path}")
        return file_path

    def extract_text_from_file(self, file_path: str, filename: str) -> str:
        """从文件中提取文本内容"""
        file_ext = os.path.splitext(filename)[1].lower()

        logger.info(f"📄 开始提取文本: {filename} ({file_ext})")

        try:
            if file_ext == '.pdf':
                logger.info("📖 使用PDF解析器提取文本")
                text = self._extract_from_pdf(file_path)
            elif file_ext == '.docx':
                logger.info("📖 使用DOCX解析器提取文本")
                text = self._extract_from_docx(file_path)
            elif file_ext == '.txt':
                logger.info("📖 使用TXT解析器提取文本")
                text = self._extract_from_txt(file_path)
            else:
                logger.error(f"❌ 不支持的文件类型: {file_ext}")
                raise ValueError(f"不支持的文件类型: {file_ext}")

            logger.info(f"✅ 文本提取成功: {len(text)} 字符")
            if text:
                preview = text[:100].replace('\n', ' ').replace('\r', ' ')
                logger.info(f"📝 文本预览: {preview}...")

            return text
        except Exception as e:
            logger.error(f"❌ 文本提取失败: {str(e)}")
            raise Exception(f"文件内容提取失败: {str(e)}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """从PDF文件提取文本 - 先尝试pypdf，失败则使用OCR"""

        # 方法1: 先尝试pypdf提取文本
        logger.info("📖 方法1: 使用pypdf提取文本")
        text = self._extract_with_pypdf(file_path)
        if text.strip():
            logger.info(f"✅ pypdf成功提取文本: {len(text)} 字符")
            return text.strip()

        # 方法2: 如果pypdf失败，使用OCR处理
        logger.info("🔍 pypdf无法提取文本，尝试OCR处理")
        text = self._extract_with_ocr(file_path)
        if text.strip():
            logger.info(f"✅ OCR成功提取文本: {len(text)} 字符")
            return text.strip()

        logger.error("❌ 所有PDF文本提取方法都失败了")
        return ""

    def _extract_with_pypdf(self, file_path: str) -> str:
        """使用pypdf提取文本"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"pypdf提取失败: {e}")
            return ""

    def _extract_with_ocr(self, file_path: str) -> str:
        """使用OCR从PDF提取文本"""
        try:
            logger.info("🔧 开始OCR处理PDF...")

            # 检查是否安装了pytesseract
            try:
                pytesseract.get_tesseract_version()
                logger.info(f"✅ Tesseract版本: {pytesseract.get_tesseract_version()}")
            except Exception as e:
                logger.error(f"❌ Tesseract未安装或配置错误: {e}")
                logger.info("💡 请安装Tesseract OCR引擎并确保在PATH中")
                return ""

            # 检查是否有fitz (PyMuPDF)用于将PDF转换为图片
            try:
                import fitz
            except ImportError:
                logger.error("❌ 需要安装PyMuPDF来将PDF转换为图片进行OCR")
                logger.info("💡 运行: pip install PyMuPDF")
                return ""

            text = ""
            doc = fitz.open(file_path)
            logger.info(f"📄 PDF页数: {len(doc)}")

            for page_num in range(len(doc)):
                logger.info(f"🔍 处理第{page_num+1}页...")
                page = doc[page_num]

                # 将PDF页面转换为图片
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 提高分辨率
                img_data = pix.tobytes("png")

                # 使用PIL打开图片
                img = Image.open(io.BytesIO(img_data))

                # 使用OCR提取文本
                try:
                    page_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                    if page_text.strip():
                        text += page_text + "\n"
                        logger.info(f"✅ 第{page_num+1}页OCR提取文本: {len(page_text)} 字符")
                    else:
                        logger.warning(f"⚠️ 第{page_num+1}页OCR未提取到文本")
                except Exception as e:
                    logger.error(f"❌ 第{page_num+1}页OCR处理失败: {e}")

            doc.close()
            logger.info(f"✅ OCR处理完成，总文本长度: {len(text)}")
            return text

        except Exception as e:
            logger.error(f"❌ OCR处理失败: {e}")
            return ""

    def _extract_from_docx(self, file_path: str) -> str:
        """从Word文档提取文本"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()

    def _extract_from_txt(self, file_path: str) -> str:
        """从文本文件提取文本"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read().strip()

    async def process_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        处理上传的文件 - 简化版，直接提取文本

        Args:
            file_content: 文件内容
            filename: 文件名

        Returns:
            处理结果字典
        """
        try:
            logger.info(f"🔄 开始处理文件: {filename}")

            # 1. 验证文件扩展名
            file_ext = os.path.splitext(filename)[1].lower()
            logger.info(f"📋 验证文件扩展名: {file_ext}")
            if file_ext not in config.ALLOWED_EXTENSIONS:
                raise ValueError(f"不支持的文件类型: {file_ext}")

            # 2. 验证文件大小
            logger.info(f"📏 验证文件大小: {len(file_content)} bytes")
            if len(file_content) > config.MAX_FILE_SIZE:
                raise ValueError(f"文件大小超过限制: {len(file_content)} bytes")

            # 3. 保存文件
            logger.info("💾 开始保存文件到磁盘...")
            file_path = await self.save_file(file_content, filename)

            # 4. 提取文本内容
            logger.info("📝 开始提取文本内容...")
            text_content = self.extract_text_from_file(file_path, filename)

            if not text_content.strip():
                logger.error("❌ 文件内容为空")
                # 检查是否是PDF文件
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext == '.pdf':
                    logger.error("❌ PDF文本提取失败（包括OCR尝试）")
                    logger.info("💡 请检查：1) Tesseract OCR引擎是否正确安装 2) PDF文件是否损坏 3) 是否为纯图片PDF且无法识别")
                    raise ValueError("无法从PDF文件中提取文本内容，OCR处理也失败了。请确保：1) Tesseract OCR引擎已正确安装 2) PDF文件不是损坏的 3) 图片质量足够高以便OCR识别")
                else:
                    raise ValueError("无法从文件中提取文本内容")

            # 5. 文件已保存，不删除以供RAG使用
            logger.info("✅ 文本提取完成，准备进行RAG向量化...")

            # 6. 将文档内容添加到向量数据库供RAG检索
            logger.info("🧠 开始RAG向量化处理...")
            vector_result = add_documents_to_vector_store(
                text_content=text_content,
                filename=filename,
                file_path=file_path
            )

            logger.info(f"🎯 RAG处理结果: {vector_result}")

            # 返回处理结果，包含RAG相关信息
            return {
                "success": True,
                "filename": filename,
                "file_path": file_path,
                "text_length": len(text_content),
                "text_content": text_content,
                "rag_info": vector_result,  # 添加RAG处理结果
                "medical_analysis": {
                    "is_medical": True,
                    "keyword_score": 1.0,
                    "embedding_score": 1.0,
                    "confidence": 1.0,
                    "details": {
                        "found_keywords": [],
                        "text_length": len(text_content),
                        "text_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
                    }
                }
            }

        except Exception as e:
            logger.error(f"❌ 文件处理异常: {str(e)}")
            import traceback
            logger.error(f"📋 异常详情: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }

    def get_uploads_status(self) -> Dict[str, Any]:
        """
        获取uploads目录状态

        Returns:
            目录状态信息字典
        """
        try:
            logger.info("📊 获取uploads目录状态")

            if not os.path.exists(config.UPLOAD_DIR):
                return {
                    "file_count": 0,
                    "files": [],
                    "total_size": "0B"
                }

            # 获取目录中的所有文件
            files = []
            total_size = 0

            for item in os.listdir(config.UPLOAD_DIR):
                item_path = os.path.join(config.UPLOAD_DIR, item)
                if os.path.isfile(item_path):
                    file_size = os.path.getsize(item_path)
                    files.append({
                        "name": item,
                        "size": file_size,
                        "size_human": self._format_size(file_size)
                    })
                    total_size += file_size

            return {
                "file_count": len(files),
                "files": [f["name"] for f in files],
                "total_size": self._format_size(total_size),
                "detailed_files": files
            }

        except Exception as e:
            logger.error(f"❌ 获取uploads目录状态失败: {e}")
            return {
                "file_count": 0,
                "files": [],
                "total_size": "0B"
            }

    def clear_uploads_directory(self) -> Dict[str, Any]:
        """
        清空uploads目录中的所有文件

        Returns:
            操作结果字典
        """
        try:
            logger.info("🗑️ 开始清空uploads目录")

            if not os.path.exists(config.UPLOAD_DIR):
                logger.warning(f"⚠️ uploads目录不存在: {config.UPLOAD_DIR}")
                return {
                    "success": True,
                    "message": "uploads目录不存在，无需清空",
                    "deleted_count": 0,
                    "deleted_files": []
                }

            # 获取目录中的所有文件
            files = []
            for item in os.listdir(config.UPLOAD_DIR):
                item_path = os.path.join(config.UPLOAD_DIR, item)
                if os.path.isfile(item_path):
                    files.append(item)

            if not files:
                logger.info("📁 uploads目录为空，无需清空")
                return {
                    "success": True,
                    "message": "uploads目录为空，无需清空",
                    "deleted_count": 0,
                    "deleted_files": []
                }

            # 删除所有文件
            deleted_files = []
            deleted_count = 0

            for file in files:
                file_path = os.path.join(config.UPLOAD_DIR, file)
                try:
                    os.remove(file_path)
                    deleted_files.append(file)
                    deleted_count += 1
                    logger.info(f"✅ 已删除文件: {file}")
                except Exception as e:
                    logger.error(f"❌ 删除文件失败 {file}: {e}")

            logger.info(f"🎉 清空uploads目录完成，共删除 {deleted_count} 个文件")

            return {
                "success": True,
                "message": f"成功清空uploads目录，删除了 {deleted_count} 个文件",
                "deleted_count": deleted_count,
                "deleted_files": deleted_files
            }

        except Exception as e:
            logger.error(f"❌ 清空uploads目录失败: {e}")
            return {
                "success": False,
                "message": f"清空uploads目录失败: {str(e)}",
                "deleted_count": 0,
                "deleted_files": []
            }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """格式化文件大小显示"""
        if size_bytes == 0:
            return "0B"

        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f}{size_names[i]}"

    @staticmethod
    def cleanup_file(file_path: str):
        """清理文件"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"清理文件失败: {e}")