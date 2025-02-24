import os
import re
import pdfplumber

def extract_all_text(pdf_path):
    """从指定的PDF文件中提取所有页面的文本，并将其作为一个字符串返回"""
    if not os.path.exists(pdf_path):
        return "指定的文件不存在，请检查路径"

    all_text = ""  # 初始化一个空字符串，用于存储所有文本
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # 依次遍历PDF文档中的每一页
            for page in pdf.pages:
                # 提取每一页的文本信息
                page_text = page.extract_text()
                all_text += page_text + "\n"
    except Exception as e:
        return f"处理 PDF 文件时出现错误: {str(e)}"
    return all_text

def extract_text_from_latex(latex_path):
    if not os.path.exists(latex_path):
        return "指定的文件不存在，请检查路径"

    all_text = ""
    try:
        with open(latex_path, 'r', encoding='utf-8') as file:
            latex_content = file.read()
            clean_text = re.sub(r'\$$a-zA-Z]+\{[^}]*\}', '', latex_content)
            clean_text = re.sub(r'\$$a-zA-Z]+', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            all_text = clean_text
    except Exception as e:
        return f"处理 LaTeX 文件时出现错误: {str(e)}"
    return all_text

#错误代码
# def split_text_by_punctuation(text):
#     pattern = r"[---------------------------------------]+"
#     segments = re.split(pattern, text)
#     return [segment for segment in segments if segments]
def split_text_by_punctuation(text):
    pattern = r"---------------------------------------"  # 固定匹配 39 个 '-'
    segments = re.split(pattern, text)
    return [segment for segment in segments if segment]  # 过滤空字符串

if __name__ == '__main__':
    all_text = extract_all_text("F:\Download\课本+自我巩固+课堂落实（答案）(1).pdf")
    print(all_text)