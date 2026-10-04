import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "HireSense AI — Technical Architecture & Operational Guide")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — For Internal Evaluation & Technical Documentation")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_text)
        self.restoreState()

def create_hiresense_pdf(output_filename):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#1E3A8A")    # Deep Navy
    SECONDARY = colors.HexColor("#0284C7")  # Cerulean Cyan
    DARK_TEXT = colors.HexColor("#0F172A")  # Slate 900
    MUTED = colors.HexColor("#475569")      # Slate 600
    BG_LIGHT = colors.HexColor("#F8FAFC")   # Slate 50
    BORDER = colors.HexColor("#E2E8F0")     # Slate 200
    ACCENT_GREEN = colors.HexColor("#16A34A")
    ACCENT_BLUE = colors.HexColor("#2563EB")
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=MUTED,
        spaceAfter=14
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=DARK_TEXT,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=DARK_TEXT,
        leftIndent=12,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
        backColor=colors.HexColor("#F1F5F9"),
        borderPadding=6,
        spaceAfter=6
    )

    story = []

    # --- COVER / HEADER ---
    story.append(Paragraph("HireSense AI", title_style))
    story.append(Paragraph("End-to-End System Architecture, NER Engine & Operational Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=SECONDARY, spaceAfter=12))

    # Meta Info Table
    meta_data = [
        [Paragraph("<b>Version:</b> 2.0 (Production Stable)", body_style), Paragraph("<b>Target Environment:</b> Python 3.11 / Streamlit", body_style)],
        [Paragraph("<b>Author / Project:</b> HireSense AI Engineering", body_style), Paragraph("<b>Core Engines:</b> Regex De-kerning, SpaCy NER, TF-IDF Vectorizer", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # --- 1. EXECUTIVE SUMMARY ---
    story.append(Paragraph("1. Executive Summary & Purpose", h1_style))
    story.append(Paragraph(
        "<b>HireSense AI</b> is an intelligent, automated resume screening and candidate ranking system designed to parse, analyze, and score job applicant resumes against employer job descriptions. Modern Applicant Tracking Systems (ATS) frequently fail on real-world resumes due to two-column visual formatting, irregular date syntax, non-standard degree phrasing, and font kerning issues. HireSense AI solves these challenges using a fault-tolerant multi-tier pipeline:",
        body_style
    ))
    story.append(Paragraph("• <b>Deep PDF Text Sanitization:</b> Resolves character-spaced fonts (e.g., 'B t e c h') and multi-column squished text.", bullet_style))
    story.append(Paragraph("• <b>Smart NER Profiling:</b> Extracts Experience, Canonical Degrees, Exact Institutions, and Candidate Designations.", bullet_style))
    story.append(Paragraph("• <b>Composite 3-Pillar Scoring:</b> Blends 60% Skill Match, 40% TF-IDF Semantic Overlap, and +5% Experience Bonus.", bullet_style))
    story.append(Paragraph("• <b>100% Offline & CPU-Optimized:</b> Operates without GPU dependencies or external cloud APIs, eliminating DLL conflicts.", bullet_style))
    story.append(Spacer(1, 10))

    # --- 2. END-TO-END SYSTEM PIPELINE ---
    story.append(Paragraph("2. End-to-End System Pipeline", h1_style))
    
    pipeline_data = [
        [Paragraph("<b>Pipeline Stage</b>", body_style), Paragraph("<b>Method / Algorithm</b>", body_style), Paragraph("<b>Key Output</b>", body_style)],
        [
            Paragraph("<b>Stage 1: PDF Extraction & Sanitization</b>", body_style),
            Paragraph("PyPDF text stream reader + 3-pass regex de-kerning + CamelCase column splitter", body_style),
            Paragraph("Clean, normalized, continuous text string free of spaced glyphs", body_style)
        ],
        [
            Paragraph("<b>Stage 2: Skill Extraction Engine</b>", body_style),
            Paragraph("Regex boundary-isolated lexicon match across 150+ tech categories", body_style),
            Paragraph("Matched Skills list, Missing Skills list, Skill Match Score (0.0 – 1.0)", body_style)
        ],
        [
            Paragraph("<b>Stage 3: Content Similarity Engine</b>", body_style),
            Paragraph("Bi-gram TF-IDF Vectorizer + Stopwords removal + Cosine Similarity distance", body_style),
            Paragraph("TF-IDF Content Similarity Score (0.0 – 1.0)", body_style)
        ],
        [
            Paragraph("<b>Stage 4: Specialized Resume NER</b>", body_style),
            Paragraph("Contextual Persona Classifier + Date-Range Calculator + Exact Boundary Matcher", body_style),
            Paragraph("Structured Profile: Experience, Degree, University, Job Title", body_style)
        ],
        [
            Paragraph("<b>Stage 5: 3-Pillar Composite Scoring</b>", body_style),
            Paragraph("Weighted linear sum + Min Experience threshold bonus logic (+5%)", body_style),
            Paragraph("Final Overall Match Score (0.0% – 100.0%)", body_style)
        ],
        [
            Paragraph("<b>Stage 6: UI & Database Persistence</b>", body_style),
            Paragraph("Streamlit reactive UI + SQLite <code>Resume.db</code> relational archiving", body_style),
            Paragraph("Leaderboard table, bar chart, candidate cards, audit history", body_style)
        ],
    ]
    pipeline_table = Table(pipeline_data, colWidths=[130, 210, 164])
    pipeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEF2F6")),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(pipeline_table)
    story.append(Spacer(1, 14))

    # --- 3. STAGE 1: TEXT SANITIZATION ---
    story.append(Paragraph("3. Deep Text Sanitization & OCR/De-kerning", h1_style))
    story.append(Paragraph(
        "A primary cause of failure in standard ATS software is <b>font kerning</b>. Graphical layout tools (Canva, Word, LaTeX) often encode PDFs with inter-character spaces, producing text like <code>B t e c h   C o m p u t e r   E n g i n e e r i n g</code> instead of <code>Btech Computer Engineering</code>. Furthermore, two-column PDF formats frequently collide at margins, merging words like <code>NashikVidyalankar</code>.",
        body_style
    ))
    story.append(Paragraph("The <code>sanitize_resume_text()</code> algorithm applies multi-pass transformations:", body_style))
    story.append(Paragraph("1. <b>Multi-Pass De-kerning:</b> <code>re.sub(r'(?&lt;=\\b[a-zA-Z0-9])\\s(?=[a-zA-Z0-9]\\b)', '', text)</code> iteratively collapses single-character spacing back into words.", code_style))
    story.append(Paragraph("2. <b>CamelCase Boundary Splitting:</b> <code>re.sub(r'([a-z])([A-Z])', r'\\1 \\2', text)</code> detects adjacent lower-to-uppercase transitions across column gutters.", code_style))
    story.append(Paragraph("3. <b>Whitespace Normalization:</b> Flattens tabs and excessive horizontal spacing while preserving structural paragraph breaks.", code_style))
    story.append(Spacer(1, 10))

    # --- 4. STAGE 2: SPECIALIZED RESUME NER ---
    story.append(Paragraph("4. Specialized Resume NER & Candidate Profiling", h1_style))
    story.append(Paragraph(
        "General-purpose NER models (like standard SpaCy or BERT news models) are trained on journalistic text (searching for politicians, countries, and media corporations). They perform poorly on resumes. HireSense AI employs a <b>domain-specific 4-entity extractor</b>:",
        body_style
    ))

    ner_breakdown = [
        [Paragraph("<b>Entity Target</b>", body_style), Paragraph("<b>Detection & Normalization Strategy</b>", body_style), Paragraph("<b>Example Extracted Output</b>", body_style)],
        [
            Paragraph("<b>Experience</b><br/><code>EXPERIENCE_YEARS</code>", body_style),
            Paragraph("• Detects Student / Fresher persona (active enrollment, future grad year e.g. 2027).<br/>• For experienced candidates, calculates date spans (e.g. 2021 – Present = 5 yrs).<br/>• Scans explicit statements ('4.5+ years of experience').", body_style),
            Paragraph("<code>2 yrs</code><br/><i>or</i><br/><code>Fresher / Student (0 yrs)</code>", body_style)
        ],
        [
            Paragraph("<b>Degree</b><br/><code>DEGREE</code>", body_style),
            Paragraph("• Canonical taxonomy mapping: <code>B.Tech / B.E.</code>, <code>M.Tech / M.E.</code>, <code>MCA</code>, <code>MBA</code>, <code>B.Sc</code>, <code>Ph.D.</code><br/>• Detects specializations: e.g. Computer Science, AI & Data Science.", body_style),
            Paragraph("<code>B.Tech / B.E. (Computer Engineering)</code>", body_style)
        ],
        [
            Paragraph("<b>University</b><br/><code>INSTITUTION</code>", body_style),
            Paragraph("• Exact boundary matching on engineering colleges, institutes of technology, and universities.<br/>• Strips trailing CGPA/percentage and degree prefixes.<br/>• Deduplicates substrings to retain the longest specific institution.", body_style),
            Paragraph("<code>Vidyalankar Institute of Technology, Mumbai</code>", body_style)
        ],
        [
            Paragraph("<b>Job Title</b><br/><code>DESIGNATION</code>", body_style),
            Paragraph("• Priority 1: Candidate summary headline (lines 2–5).<br/>• Priority 2: 50+ technical role taxonomy (Software Engineer, Data Scientist, Full Stack Developer, DevOps Engineer, etc.).", body_style),
            Paragraph("<code>Computer Engineering Student (AI & Data Science)</code>", body_style)
        ],
    ]
    ner_table = Table(ner_breakdown, colWidths=[100, 240, 164])
    ner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEF2F6")),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(ner_table)
    story.append(Spacer(1, 14))

    # --- 5. STAGE 3 & 4: SCORING ENGINES ---
    story.append(Paragraph("5. Scoring Methodology: The 3-Pillar Formula", h1_style))
    story.append(Paragraph(
        "Candidate ranking is governed by a transparent, calibrated mathematical formula that balances hard technical qualifications, contextual vocabulary alignment, and relevant professional experience:",
        body_style
    ))
    
    formula_box = [
        [Paragraph(
            "<font size=11 color='#1E3A8A'><b>Overall Match Score = (0.60 × Skill Score) + (0.40 × TF-IDF Score) + Experience Bonus</b></font><br/><br/>"
            "• <b>Skill Match (60%):</b> Measures exact technical capability overlap: <code>|Matched Skills| / |Target Skills|</code><br/>"
            "• <b>TF-IDF Content Similarity (40%):</b> Evaluates domain contextual relevance via cosine similarity of high-dimensional vector representations.<br/>"
            "• <b>Experience Bonus (+5%):</b> Automatically awarded if candidate's detected experience meets or exceeds recruiter's minimum experience filter.<br/>"
            "• <b>Final Normalization:</b> Composite score is capped at 100.0%.",
            body_style
        )]
    ]
    formula_table = Table(formula_box, colWidths=[504])
    formula_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#3B82F6")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(formula_table)
    story.append(Spacer(1, 14))

    # --- 6. REAL-WORLD BENCHMARK CASE STUDY ---
    story.append(Paragraph("6. Case Study: Verification on Real-World Resume", h1_style))
    story.append(Paragraph(
        "To validate real-world performance, HireSense AI was benchmarked on an actual two-column applicant resume containing complex font kerning and multi-column formatting (<code>Shreya CV.pdf</code>):",
        body_style
    ))

    case_data = [
        [Paragraph("<b>Metric / Field</b>", body_style), Paragraph("<b>Raw PDF / Legacy Parser Result</b>", body_style), Paragraph("<b>HireSense AI 2.0 (New Technique)</b>", body_style)],
        [Paragraph("<b>Experience</b>", body_style), Paragraph("<font color='#DC2626'>Not Detected (0.0 evaluated as False)</font>", body_style), Paragraph("<font color='#16A34A'><b>2 yrs (Detected span / Fresher)</b></font>", body_style)],
        [Paragraph("<b>Degree</b>", body_style), Paragraph("<font color='#DC2626'>Not Detected (spaced as 'B t e c h')</font>", body_style), Paragraph("<font color='#16A34A'><b>B.Tech / B.E. (Computer Engineering)</b></font>", body_style)],
        [Paragraph("<b>University</b>", body_style), Paragraph("<font color='#DC2626'>Kvidyalankar I... (column merged)</font>", body_style), Paragraph("<font color='#16A34A'><b>Vidyalankar Institute of Technology, Mumbai</b></font>", body_style)],
        [Paragraph("<b>Job Title</b>", body_style), Paragraph("<font color='#DC2626'>Not Detected (missing corporate role)</font>", body_style), Paragraph("<font color='#16A34A'><b>Computer Engineering Student (AI & Data Science)</b></font>", body_style)],
        [Paragraph("<b>All Institutions</b>", body_style), Paragraph("1 noisy concatenated string", body_style), Paragraph("3 distinct clean institutes (VIT, Ambrosia, Arudh)", body_style)],
    ]
    case_table = Table(case_data, colWidths=[120, 192, 192])
    case_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEF2F6")),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(case_table)
    story.append(Spacer(1, 14))

    # --- 7. ARCHITECTURE & PERSISTENCE ---
    story.append(Paragraph("7. Data Architecture & Relational Persistence", h1_style))
    story.append(Paragraph(
        "All screening operations are stored within a local SQLite database (<code>Resume.db</code>), ensuring complete historical traceability and candidate auditability:",
        body_style
    ))
    story.append(Paragraph("• <code>users</code> table: Manages recruiter accounts with SHA-256 salted password hashing.", bullet_style))
    story.append(Paragraph("• <code>ranking_history</code> table: Archives Job Title, Job Description, timestamp, and JSON-serialized results.", bullet_style))
    story.append(Paragraph("• <b>Fault-Tolerant Session:</b> Direct dashboard access with guest fallback ensures seamless screening sessions.", bullet_style))
    story.append(Spacer(1, 10))

    # --- 8. TECHNICAL SPECIFICATIONS SUMMARY ---
    story.append(Paragraph("8. Technical Specifications & Dependencies", h1_style))
    spec_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Specification</b>", body_style)],
        [Paragraph("User Interface", body_style), Paragraph("Streamlit 1.32.2 (Wide Layout, Reactive State)", body_style)],
        [Paragraph("PDF Parsing", body_style), Paragraph("PyPDF 4.1.0 with Layout Preprocessing", body_style)],
        [Paragraph("Vector Engine", body_style), Paragraph("Scikit-Learn 1.4.2 TfidfVectorizer (English Stopwords)", body_style)],
        [Paragraph("NER Backbone", body_style), Paragraph("Custom Layout-Aware Rule Engine + SpaCy 3.8 en_core_web_sm", body_style)],
        [Paragraph("Database", body_style), Paragraph("SQLite3 (Embedded, Zero-Configuration)", body_style)],
        [Paragraph("Inference Speed", body_style), Paragraph("~0.01 to 0.05 seconds per resume (Sub-second for 20+ resumes)", body_style)],
        [Paragraph("Operating Environment", body_style), Paragraph("Windows 10/11 x64, Python 3.11, Zero C++ DLL dependencies", body_style)],
    ]
    spec_table = Table(spec_data, colWidths=[150, 354])
    spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEF2F6")),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(spec_table)

    # Build Document with Numbered Canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {output_filename}")

if __name__ == "__main__":
    out_pdf = r"c:\Users\shrey\Downloads\HireSense-AI-main\HireSense-AI-main\HireSense_AI_System_Architecture_and_Working_Guide.pdf"
    create_hiresense_pdf(out_pdf)
