# -*- coding: utf-8 -*-
"""
build_resume.py — 生成许家成的一页式中文简历（DOCX）
运行环境：Codex 捆绑 Python（含 python-docx）
用法： python scripts/build_resume.py
输出： resume/许家成-简历-储能数据分析实习.docx
"""
from __future__ import annotations
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "resume"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "许家成-简历-储能数据分析实习.docx"

CN_FONT = "微软雅黑"
EN_FONT = "Calibri"
BODY_SIZE = Pt(10.3)
SMALL_SIZE = Pt(9.3)

# ---------------- 内容区（改这里就能更新简历） ----------------
NAME = "许家成"
CONTACT = "19996882586　|　3532858925@qq.com　|　福州"
INTENT = "求职意向：储能数据分析 / 电池测试 / 运维数据 实习生"
AVAILABLE = "可到岗：寒假 1.20 起全职至假期结束；开学后每周可到岗 2-3 天"

EDUCATION = [
    ("福建技术师范学院　|　新能源科学与工程　|　本科 · 大二　|　2025.09 – 2029.06（预计）", True),
    ("相关课程：电池原理、电路、自动控制、Python 程序设计与数据分析（按实际调整）", False),
    ("GPA / 排名：（如有优势请填写；无优势可删除本行）", False),
]

PROJECT_TITLE = "储能电芯一致性巡检与热失控早期预警工具　|　个人项目　|　2026.08 – 至今"
PROJECT_STACK = "技术栈：Python · pandas · scikit-learn · Streamlit · ReportLab · Git"
PROJECT_BULLETS = [
    ("数据工程", "基于 NASA PCoE（174 万行）与马里兰大学 CALCE（64.7 万行）两套真实数据构建多电芯同步时间轴与格式适配层；实现自适应断点阈值，解决 30 秒采样被误判为通讯中断、53 万行数据失效的问题。"),
    ("清洗与特征", "断点分级处理（≤3s 线性插值、>5s 判定通讯离线且不伪造）、中值滤波去除毛刺、充放电/静置工况切分；提取温升速率、电压一致性偏差、组内极差、静置自松弛斜率等 7 项物理特征。"),
    ("双引擎判定", "物理规则（逐电芯×工况的动态 3σ 包络，二级预警由机理判据独立触发）与 IsolationForest（发现新型异常）协同；修复阈值随批次浮动的数据泄露问题，改为健康基线固化。"),
    ("实验验证", "完成 5 组消融实验 × 2 场景：健康数据误报由 5.01% 降至 1.31%；注入热失控前兆 0.5 分钟检出（单用 IsolationForest 滞后 102 分钟）；注入 −15 mV/min 微短路验证静置斜率特征，信噪比约 70 倍。"),
    ("交付形态", "Streamlit 可视化大屏、JSON/Markdown/PDF 诊断报告、命令行 Skill 接口；19 项 pytest 全部通过，Git 规范化提交。"),
]

SKILLS = [
    ("编程与数据", "Python（pandas / numpy / scikit-learn）、时序数据清洗与特征工程、数据可视化"),
    ("工具", "Excel、Git/GitHub、Streamlit、VS Code"),
    ("电池基础", "充放电循环、容量/SOH、直流内阻、温度特性、电芯一致性、热失控机理"),
]

OTHERS = [
    "海外运营 / 跨文化沟通：（请填写具体平台、职责与成果，如内容运营、客户沟通、订单跟进等）",
    "竞赛 / 证书 / 奖学金：（如有请填写，无则删除本行）",
]

SUMMARY = "新能源专业在读，对储能电池数据分析有持续兴趣；具备从数据清洗、特征构建到模型验证的完整实践，习惯用对照实验与消融分析验证结论，并能主动说明方法的适用边界。"


# ---------------- 样式辅助 ----------------
def set_run_font(run, size=BODY_SIZE, bold=False, color=(0, 0, 0)):
    run.font.size = size
    run.font.bold = bold
    run.font.name = EN_FONT
    run.font.color.rgb = RGBColor(*color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), CN_FONT)
    rfonts.set(qn("w:ascii"), EN_FONT)
    rfonts.set(qn("w:hAnsi"), EN_FONT)


def para(space_before=0, space_after=0, line=1.12, align=None, indent=None, hanging=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if align is not None:
        p.alignment = align
    if indent is not None:
        pf.left_indent = Cm(indent)
    if hanging is not None:
        pf.first_line_indent = Cm(-hanging)
    return p


def section_heading(text):
    p = para(space_before=9.5, space_after=3.5, line=1.0)
    r = p.add_run(text)
    set_run_font(r, size=Pt(11), bold=True)
    # 底部细线
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), "808080")
    bottom.set(qn("w:space"), "2")
    pbdr.append(bottom)
    pPr.append(pbdr)
    return p


def bullet(label, text):
    p = para(space_before=2.0, space_after=2.0, line=1.15, indent=0.42, hanging=0.42)
    r0 = p.add_run("• ")
    set_run_font(r0, size=BODY_SIZE)
    if label:
        r1 = p.add_run(label + "：")
        set_run_font(r1, size=BODY_SIZE, bold=True)
    r2 = p.add_run(text)
    set_run_font(r2, size=BODY_SIZE)
    return p


# ---------------- 文档 ----------------
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(1.35)
sec.left_margin = sec.right_margin = Cm(1.6)

style = doc.styles["Normal"]
style.font.name = EN_FONT
style.font.size = BODY_SIZE
style.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

# 姓名
p = para(space_after=1, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
r = p.add_run(NAME)
set_run_font(r, size=Pt(22), bold=True)

# 联系方式
p = para(space_after=1, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
r = p.add_run(CONTACT)
set_run_font(r, size=Pt(9.2))

# 求职意向 + 到岗时间
p = para(space_after=0.5, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
r = p.add_run(INTENT)
set_run_font(r, size=Pt(9.2), bold=True)
p = para(space_after=2, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
r = p.add_run(AVAILABLE)
set_run_font(r, size=SMALL_SIZE, color=(60, 60, 60))

# 教育背景
section_heading("教育背景")
for text, bold in EDUCATION:
    p = para(space_before=0.6, space_after=0.6, line=1.08)
    r = p.add_run(text)
    set_run_font(r, size=BODY_SIZE, bold=bold)

# 项目经历
section_heading("项目经历")
p = para(space_before=0.6, space_after=0.6, line=1.08)
r = p.add_run(PROJECT_TITLE)
set_run_font(r, size=BODY_SIZE, bold=True)
p = para(space_after=1.2, line=1.05)
r = p.add_run(PROJECT_STACK)
set_run_font(r, size=SMALL_SIZE, color=(70, 70, 70))
for label, text in PROJECT_BULLETS:
    bullet(label, text)

# 技能
section_heading("技能")
for label, text in SKILLS:
    bullet(label, text)

# 其他经历
section_heading("其他经历")
for text in OTHERS:
    bullet("", text)

# 自我评价
section_heading("自我评价")
p = para(space_before=0.8, space_after=0, line=1.08)
r = p.add_run(SUMMARY)
set_run_font(r, size=BODY_SIZE)

doc.save(OUT_PATH)
print("SAVED:", OUT_PATH)
