from fpdf import FPDF, XPos, YPos
from utils import sanitize_for_helvetica

class BasePDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "EEG Analysis Report"
        self.primary_color = (52, 73, 94)
        self.secondary_color = (74, 144, 226)
        self.text_color_dark = (30, 30, 30)
        self.text_color_light = (100, 100, 100)
        self.text_color_normal = (0,0,0)
        self.line_color = (220, 220, 220)
        self.card_bg_color = (248, 249, 250)
        self.highlight_color_alz = (220, 60, 60)
        self.highlight_color_norm = (60, 179, 113)
        self.warning_bg_color = (255, 243, 205)
        self.warning_text_color = (133, 100, 4)
        self.page_margin = 15
        self.set_auto_page_break(auto=True, margin=self.page_margin)
        self.set_line_width(0.2)

    def _is_bold_font(self):
        return 'B' in self.font_style

    def cell(self, w, h=0, txt="", border=0, ln=0, align="", fill=False, link=""):
        txt_to_render = sanitize_for_helvetica(txt)
        super().cell(w, h, txt_to_render, border, ln, align, fill, link)

    def multi_cell(self, w, h, txt="", border=0, align="J", fill=False, max_line_height=0, new_x=XPos.START, new_y=YPos.TOP):
        txt_to_render = sanitize_for_helvetica(txt)
        if max_line_height == 0: max_line_height = h
        super().multi_cell(w, h, txt_to_render, border, align, fill, max_line_height=max_line_height, new_x=new_x, new_y=new_y)

    def write(self, h, txt="", link=""):
        txt_to_render = sanitize_for_helvetica(txt)
        super().write(h, txt_to_render, link)

    def header(self):
        try:
            self.set_font('Helvetica', 'B', 15)
            title = sanitize_for_helvetica(self.report_title)
            title_w = self.get_string_width(title) + 6
            doc_w = self.w
            self.set_x((doc_w - title_w) / 2)
            self.set_text_color(*self.secondary_color)
            self.cell(title_w, 10, title, border=0, align='C', ln=1)
            self.set_text_color(*self.text_color_normal)
            self.ln(5)
            self.set_draw_color(*self.line_color)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(8)
        except Exception as e:
            print(f"PDF Header Error: {e}")

    def footer(self):
        try:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
            self.set_text_color(*self.text_color_normal)
        except Exception as e:
            print(f"PDF Footer Error: {e}")

    def section_title(self, title_text: str):
        try:
            self.set_font('Helvetica', 'B', 13)
            self.set_fill_color(80, 227, 194)
            self.set_text_color(10, 15, 26)
            self.cell(0, 8, " " + sanitize_for_helvetica(title_text), border='B', align='L', fill=True, ln=1)
            self.set_text_color(*self.text_color_normal)
            self.ln(6)
        except Exception as e:
            print(f"PDF Section Title Error for '{title_text}': {e}")

    def key_value_pair(self, key: str, value, key_width=45):
        try:
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(*self.text_color_dark)
            key_start_y = self.get_y()
            self.multi_cell(key_width, 6, sanitize_for_helvetica(str(key))+":", align='L', new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=self.font_size)
            self.set_y(key_start_y)
            self.set_x(self.l_margin + key_width + 2)
            self.set_font('Helvetica', '', 10)
            self.set_text_color(*self.text_color_normal)
            self.multi_cell(0, 6, sanitize_for_helvetica(str(value)), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size)
            self.ln(1)
        except Exception as e:
            print(f"PDF Key/Value Error for key '{key}': {e}")

    def write_multiline(self, text: str, height=5, indent=5):
        try:
            self.set_font('Helvetica', '', 10)
            self.set_text_color(80, 80, 80)
            self.set_left_margin(self.l_margin + indent)
            self.multi_cell(0, height, sanitize_for_helvetica(text), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size)
            self.set_left_margin(self.l_margin)
            self.ln(height / 2)
            self.set_text_color(*self.text_color_normal)
        except Exception as e:
            print(f"PDF Multiline Error: {e}")

    def metric_card(self, title: str, value, unit: str = "", description: str = ""):
        try:
            start_x = self.get_x()
            start_y = self.get_y()
            card_width = (self.w - self.l_margin - self.r_margin - 5) / 2
            card_height = 25
            self.set_fill_color(240, 245, 250)
            self.set_draw_color(80, 227, 194)
            self.set_line_width(0.3)
            self.rect(start_x, start_y, card_width, card_height, 'DF')
            self.set_xy(start_x + 3, start_y + 3)
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(80, 80, 80)
            self.cell(card_width - 6, 5, sanitize_for_helvetica(title.upper()), align='L')
            self.set_xy(start_x + 3, start_y + 9)
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(*self.secondary_color)
            value_str = f"{sanitize_for_helvetica(str(value))}{sanitize_for_helvetica(str(unit))}"
            self.cell(card_width - 6, 8, value_str, align='R')
            if description:
                self.set_xy(start_x + 3, start_y + 18)
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(*self.text_color_light)
                self.cell(card_width - 6, 5, sanitize_for_helvetica(description), align='L')
            self.set_y(start_y)
            self.set_x(start_x + card_width + 5)
            self.set_text_color(*self.text_color_normal)
            self.set_line_width(0.2)
        except Exception as e:
            print(f"PDF Metric Card Error for title '{title}': {e}")

    def write_paragraph(self, text, height=4, indent=0, font_style='', font_size=8.5, text_color=None, bullet_char_override=None):
        try:
             self.set_font('Helvetica', font_style, font_size)
             current_text_color = text_color if text_color else self.text_color_dark
             self.set_text_color(*current_text_color)
             current_x_start = self.l_margin + indent
             self.set_x(current_x_start)
             sanitized_text = sanitize_for_helvetica(text)
             if bullet_char_override:
                 safe_bullet = sanitize_for_helvetica(bullet_char_override)
                 original_font_family, original_font_size, original_font_style = self.font_family, self.font_size_pt, self.font_style
                 self.set_font('Helvetica', 'B', font_size)
                 self.cell(self.get_string_width(safe_bullet) + 0.5, height, safe_bullet, ln=0)
                 self.set_font(original_font_family, original_font_style, original_font_size)
                 self.set_x(current_x_start + self.get_string_width(safe_bullet) + 1.5)
                 self.multi_cell(self.w - self.get_x() - self.r_margin, height, sanitized_text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size)
             else:
                 self.multi_cell(0, height, sanitized_text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size)
             self.ln(height / 4)
             self.set_text_color(*self.text_color_normal)
        except Exception as e:
            print(f"PDF write_paragraph Error: {e}")

    def add_image_section(self, title: str, image_data_base64: str):
        import base64
        import io
        image_height_estimate = 70
        title_height_estimate = 10 if title else 0
        if self.get_y() + title_height_estimate + image_height_estimate > self.h - self.b_margin:
            self.add_page()
        if title:
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(*self.text_color_dark)
            self.cell(0, 8, sanitize_for_helvetica(title), ln=1, align='L')
            self.ln(1)
        if image_data_base64 and isinstance(image_data_base64, str) and image_data_base64.startswith('data:image/png;base64,'):
            try:
                img_bytes = base64.b64decode(image_data_base64.split(',', 1)[1])
                img_file = io.BytesIO(img_bytes)
                page_content_width = self.w - 2 * self.page_margin
                img_display_width = page_content_width * 0.95
                x_pos = self.l_margin + (page_content_width - img_display_width) / 2
                self.image(img_file, x=x_pos, w=img_display_width)
                img_file.close()
                self.ln(2)
            except Exception as e:
                error_text = f"(Error embedding image '{sanitize_for_helvetica(title)}': {sanitize_for_helvetica(str(e)[:50])})"
                self.write_paragraph(error_text, font_style='I')
                print(f"PDF Image Embed Error for '{title}': {e}")
        else:
            if title:
                 self.write_paragraph(sanitize_for_helvetica("(Image data not available)"), font_style='I', indent=5)
        self.ln(4)

    def add_explanation_box(self, title: str, text_lines: list, icon_char: str = "[i]",
                            bg_color=None, title_color=None, text_color_override=None,
                            font_size_text=9, line_h=5):
        self.ln(1)
        current_bg_color = bg_color if bg_color else self.card_bg_color
        current_title_color = title_color if title_color else self.primary_color
        current_text_color = text_color_override if text_color_override else self.text_color_dark
        safe_icon = sanitize_for_helvetica(icon_char)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*current_title_color)
        title_to_render = f"{safe_icon} {sanitize_for_helvetica(title)}" if safe_icon else sanitize_for_helvetica(title)
        self.multi_cell(0, 7, title_to_render, new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size)
        y_before_text_content = self.get_y()
        estimated_box_height = 3
        for item in text_lines:
            estimated_box_height += line_h + 1
        self.set_fill_color(*current_bg_color)
        self.set_draw_color(*self.line_color)
        self.rect(self.l_margin, y_before_text_content, self.w - self.l_margin - self.r_margin, estimated_box_height, 'DF')
        self.set_y(y_before_text_content + 1.5)
        for item in text_lines:
            self.set_x(self.l_margin + 2)
            is_list_item = isinstance(item, tuple) and item[0] == "bullet"
            actual_text = item[1] if is_list_item else item
            is_sub_list_item = isinstance(actual_text, tuple) and actual_text[0] == "sub_bullet"
            final_text_content = actual_text[1] if is_sub_list_item else actual_text
            bullet_char_to_use = "-"
            if is_list_item:
                item_x_start = self.get_x()
                self.set_font('Helvetica', 'B', font_size_text)
                self.set_text_color(*current_text_color)
                if is_sub_list_item:
                    self.set_x(item_x_start + 5)
                self.cell(5, line_h, bullet_char_to_use, ln=0)
                self.set_x(item_x_start + 5 + (5 if is_sub_list_item else 0))
            parts = sanitize_for_helvetica(final_text_content).split("**")
            for i, part in enumerate(parts):
                is_bold_part = (i % 2 == 1)
                self.set_font('Helvetica', 'B' if is_bold_part else '', font_size_text)
                self.set_text_color(*(self.primary_color if is_bold_part else current_text_color))
                self.write(line_h, part)
            self.ln(line_h + 0.5)
        self.set_y(y_before_text_content + estimated_box_height)
        self.ln(3)
        self.set_text_color(*self.text_color_normal)
