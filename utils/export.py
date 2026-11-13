"""
utils/export.py
~~~~~~~~~~~~~~~
Export and reporting utilities for the Urban Tree Health Monitor.
Produces structured JSON assessment reports and formatted text summaries.
"""

import json
from datetime import datetime


def generate_json_report(filename, image_shape, index_name, stats, classification, health_percent):
    """
    Builds a structured dictionary containing all metadata, statistics,
    and diagnostic recommendations for GIS / arborist export.
    """
    report = {
        "report_metadata": {
            "application": "Urban Tree Health Monitor",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "analyzed_filename": filename,
            "image_dimensions": {
                "width": int(image_shape[1]),
                "height": int(image_shape[0]),
                "total_pixels": int(image_shape[0] * image_shape[1]),
            },
        },
        "analysis_configuration": {
            "selected_index": index_name,
            "methodology": (
                "RGB Visible-spectrum index"
                if "VARI" in index_name or "GLI" in index_name
                else "Simulated Near-Infrared NDVI"
            ),
        },
        "health_assessment": {
            "status": classification.get("status"),
            "confidence": classification.get("confidence"),
            "diagnosis": classification.get("description"),
            "vegetation_coverage_pct": stats.get("vegetation_percentage"),
            "vegetation_pixels": stats.get("vegetation_pixels"),
        },
        "pixel_health_breakdown": health_percent,
        "statistical_metrics": {
            "mean_index_value": stats.get("mean"),
            "std_deviation": stats.get("std"),
            "min_value": stats.get("min"),
            "max_value": stats.get("max"),
        },
        "recommendations": _get_recommendations(classification.get("status")),
    }
    return report


def _get_recommendations(status):
    if status == "Healthy":
        return [
            "Canopy exhibits robust photosynthetic density and healthy leaf area index.",
            "Continue standard seasonal monitoring and irrigation schedules.",
        ]
    elif status == "Moderate":
        return [
            "Early stress indicators or canopy thinning detected.",
            "Check soil moisture levels and test for nutrient deficiencies (e.g. nitrogen or iron chlorosis).",
            "Inspect foliage for localized fungal spotting or pest infestations.",
        ]
    elif status == "Stressed":
        return [
            "Critical foliage stress, extensive dieback, or senescence observed.",
            "Conduct on-site arborist inspection of root collar and main stem.",
            "Evaluate irrigation infrastructure for drought stress or waterlogging.",
            "Prune dead wood to prevent secondary pathogen spread.",
        ]
    else:
        return [
            "Insufficient foliage detected to establish vegetative health metrics.",
            "Ensure the camera frame captures at least 20% living canopy or branches.",
        ]


def format_report_as_text(report_dict):
    """Formats report dictionary into human-readable text."""
    meta = report_dict["report_metadata"]
    assess = report_dict["health_assessment"]
    breakdown = report_dict["pixel_health_breakdown"]
    stats = report_dict["statistical_metrics"]

    lines = [
        "=" * 60,
        "       URBAN TREE HEALTH ASSESSMENT REPORT",
        "=" * 60,
        f"Timestamp         : {meta['timestamp']}",
        f"File Analyzed     : {meta['analyzed_filename']}",
        f"Dimensions        : {meta['image_dimensions']['width']}x{meta['image_dimensions']['height']} px",
        f"Spectral Metric   : {report_dict['analysis_configuration']['selected_index']}",
        "-" * 60,
        "OVERALL ASSESSMENT",
        "-" * 60,
        f"Health Status     : {assess['status']}",
        f"Confidence        : {assess['confidence']}",
        f"Canopy Coverage   : {assess['vegetation_coverage_pct']:.1f}% ({assess['vegetation_pixels']:,} px)",
        f"Diagnostic Note   : {assess['diagnosis']}",
        "-" * 60,
        "CANOPY COMPOSITION",
        "-" * 60,
        f"Healthy Foliage   : {breakdown.get('Healthy', 0.0)}%",
        f"Moderate Foliage  : {breakdown.get('Moderate', 0.0)}%",
        f"Stressed Foliage  : {breakdown.get('Stressed', 0.0)}%",
        "-" * 60,
        "NUMERICAL STATISTICS",
        "-" * 60,
        f"Mean Index        : {stats['mean_index_value']:.4f}",
        f"Standard Dev      : {stats['std_deviation']:.4f}",
        f"Min / Max         : {stats['min_value']:.4f} / {stats['max_value']:.4f}",
        "-" * 60,
        "MANAGEMENT RECOMMENDATIONS",
        "-" * 60,
    ]
    for r in report_dict["recommendations"]:
        lines.append(f" - {r}")
    lines.append("=" * 60)
    return "\n".join(lines)
