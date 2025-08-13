import base64
import io
import pandas as pd
import traceback
from fpdf import XPos, YPos
from .base_report import BasePDFReport
from utils import sanitize_for_helvetica

class TechnicalPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Technical EEG Analysis Report"

def format_metric_for_pdf(value, type='float', precision=1):
    if value is None or (isinstance(value, float) and (pd.isna(value) or not pd.Series(value).notna().all())):
        return 'N/A'
    try:
        if type == 'percent':
            return f"{float(value) * 100:.{precision}f}%"
        if type == 'float':
            return f"{float(value):.{precision}f}"
        return sanitize_for_helvetica(str(value))
    except (ValueError, TypeError):
        return 'N/A'

def build_technical_pdf_report_content(pdf: TechnicalPDFReport, prediction_data, stats_data,
                                       similarity_data, consistency_metrics,
                                       ts_img_data, psd_img_data, similarity_plot_data):
    page_width = pdf.w - pdf.l_margin - pdf.r_margin

    try:
        pdf.add_page()
        pdf.section_title("Analysis Details")
        pdf.key_value_pair("Filename", prediction_data.get('filename', 'N/A'))

        created_at = prediction_data.get('created_at')
        date_str = 'N/A'
        if created_at:
            try:
                dt_obj = pd.to_datetime(created_at)
                date_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S UTC') if dt_obj.tzinfo else dt_obj.strftime('%Y-%m-%d %H:%M:%S (Unknown TZ)')
            except Exception:
                date_str = str(created_at)
        pdf.key_value_pair("Analyzed On", date_str)
        pdf.ln(5)

        pdf.section_title("ML Prediction & Internal Consistency")
        prediction_label = prediction_data.get('prediction', 'N/A')
        pdf.key_value_pair("Overall Prediction", prediction_label)

        probabilities = prediction_data.get('probabilities')
        prob_str = 'N/A'
        if isinstance(probabilities, list) and len(probabilities) == 2:
            try:
                prob_str = f"Normal: {format_metric_for_pdf(probabilities[0], 'percent', 1)}, Alzheimer's: {format_metric_for_pdf(probabilities[1], 'percent', 1)}"
            except Exception:
                prob_str = sanitize_for_helvetica(str(probabilities))
        elif probabilities is not None:
            prob_str = sanitize_for_helvetica(str(probabilities))
        pdf.key_value_pair("Confidence (Initial Segment)", prob_str)
        pdf.ln(5)

        if consistency_metrics and not consistency_metrics.get('error') and consistency_metrics.get('num_trials', 0) > 0:
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 6, "Internal Consistency Metrics:", ln=1)
            pdf.ln(2)
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 5, "(Compares segment predictions against the overall prediction for this file. Reflects model stability on this sample, not diagnostic accuracy against external ground truth.)", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            pdf.set_text_color(*pdf.text_color_normal)
            pdf.ln(2)

            metrics = consistency_metrics
            col_width = page_width / 2 - 2

            def add_metric_row(metric1_args, metric2_args=None):
                current_y = pdf.get_y()
                pdf.metric_card(*metric1_args)
                if metric2_args:
                    pdf.set_y(current_y)
                    pdf.metric_card(*metric2_args)
                pdf.ln(28)

            add_metric_row(
                ("Accuracy", format_metric_for_pdf(metrics.get('accuracy'), 'percent', 1), "", "Overall segment agreement"),
                ("Precision (Alzheimer's)", format_metric_for_pdf(metrics.get('precision'), 'float', 3), "", "TP / (TP+FP) for Alz class")
            )
            add_metric_row(
                ("Recall/Sensitivity (Alzheimer's)", format_metric_for_pdf(metrics.get('recall_sensitivity'), 'float', 3), "", "TP / (TP+FN) for Alz class"),
                ("Specificity (Normal)", format_metric_for_pdf(metrics.get('specificity'), 'float', 3), "", "TN / (TN+FP) for Normal class")
            )
            add_metric_row(
                ("F1-Score (Alzheimer's)", format_metric_for_pdf(metrics.get('f1_score'), 'float', 3), "", "Harmonic mean (Precision & Recall)"),
                ("Segments Analyzed", str(metrics.get('num_trials', 'N/A')), "", "Number of EEG segments processed")
            )

            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(100, 100, 100)
            cm_ref = metrics.get('majority_label_used_as_reference', '?')
            cm_ref_label = "Alzheimer's" if cm_ref == 1 else "Normal" if cm_ref == 0 else "?"
            conf_matrix_str = (
                f"(Confusion Matrix Ref: '{cm_ref_label}') "
                f"TP:{metrics.get('true_positives','?')} | TN:{metrics.get('true_negatives','?')} | "
                f"FP:{metrics.get('false_positives','?')} | FN:{metrics.get('false_negatives','?')}"
            )
            pdf.multi_cell(0, 5, conf_matrix_str, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*pdf.text_color_normal)
        elif consistency_metrics and consistency_metrics.get('message'):
            pdf.set_font('Helvetica', 'I', 10)
            pdf.write_multiline(f"(Consistency: {consistency_metrics['message']})", indent=5)
        else:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.write_multiline("(Internal consistency metrics not calculated or not applicable.)", indent=5)
        pdf.ln(5)

        if pdf.get_y() > pdf.h - 80:
            pdf.add_page()

        pdf.section_title("Signal Shape Similarity Analysis (DTW)")
        if similarity_data and not similarity_data.get('error'):
            pdf.write_multiline(similarity_data.get('interpretation', 'No similarity interpretation available.'), indent=5)
            pdf.ln(2)
            if similarity_plot_data:
                plotted_ch_idx = similarity_data.get('plotted_channel_index')
                plot_title_sim = f"Channel {plotted_ch_idx + 1 if plotted_ch_idx is not None else '?'} Comparison Plot:"
                pdf.add_image_section(plot_title_sim, similarity_plot_data)
            else:
                pdf.set_font("Helvetica",'I',10)
                pdf.cell(0,10,"(Similarity plot not generated or invalid)", ln=1)
        else:
            err_msg_sim = similarity_data.get('error', 'Unknown') if similarity_data else 'data not available'
            pdf.set_font("Helvetica",'I',10)
            pdf.write_multiline(f"(Similarity Analysis Error: {err_msg_sim})", indent=5)
        pdf.ln(5)

        if pdf.get_y() > pdf.h - 60:
            pdf.add_page()

        pdf.section_title("Descriptive Statistics")
        if stats_data and not stats_data.get('error'):
            pdf.set_font("Helvetica",'B',11)
            pdf.cell(0,6,"Average Relative Band Power (%):", ln=1)
            pdf.ln(1)
            pdf.set_font("Helvetica",'',10)
            avg_power = stats_data.get('avg_band_power',{})
            band_found=False
            if avg_power:
                for band, powers in avg_power.items():
                    rel_power = powers.get('relative')
                    band_found |= (rel_power is not None)
                    pdf.key_value_pair(f"- {band.capitalize()}", format_metric_for_pdf(rel_power, 'percent', 2), key_width=35)
            if not band_found:
                pdf.set_font("Helvetica",'I',10)
                pdf.cell(10)
                pdf.cell(0,5,"(No band power data)", ln=1)
            pdf.ln(3)

            std_devs = stats_data.get('std_dev_per_channel')
            if std_devs:
                pdf.set_font("Helvetica",'B',11)
                pdf.cell(0,6,"Standard Deviation per Channel (µV):", ln=1)
                pdf.ln(1)
                std_dev_str = ", ".join([format_metric_for_pdf(s, 'float', 2) for s in std_devs])
                pdf.set_font("Helvetica",'',9)
                pdf.write_multiline(std_dev_str, indent=2)
        else:
            err_msg_stats = stats_data.get('error', 'Unknown') if stats_data else 'data not available'
            pdf.set_font("Helvetica",'I',10)
            pdf.write_multiline(f"(Statistics Error: {err_msg_stats})", indent=5)
        pdf.ln(5)

        if pdf.get_y() > pdf.h - 100:
            pdf.add_page()
        pdf.add_image_section("Stacked Time Series", ts_img_data)

        if pdf.get_y() > pdf.h - 100:
            pdf.add_page()
        pdf.add_image_section("Average Power Spectral Density (PSD)", psd_img_data)

        pdf.ln(10)
        if pdf.get_y() > pdf.h - 30:
            pdf.add_page()
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(128,128,128)
        pdf.multi_cell(0, 5, "This AI-driven report is for informational and technical review purposes. It is not a substitute for professional medical diagnosis. All interpretations require clinical correlation.", align='C')
        pdf.set_text_color(*pdf.text_color_normal)

    except Exception as pdf_build_e:
        print(f"Error building Technical PDF content: {pdf_build_e}")
        traceback.print_exc()
        try:
            if pdf.page_no() == 0:
                pdf.add_page()
            elif pdf.get_y() > pdf.h - 30:
                pdf.add_page()
            pdf.set_font("Helvetica",'B',12)
            pdf.set_text_color(255,0,0)
            pdf.multi_cell(0,10,f"Critical Error Building PDF Content:\n{sanitize_for_helvetica(str(pdf_build_e))}",align='C')
            pdf.set_text_color(*pdf.text_color_normal)
        except Exception as pdf_err_fallback:
            print(f"Fallback error writing to Technical PDF failed: {pdf_err_fallback}")