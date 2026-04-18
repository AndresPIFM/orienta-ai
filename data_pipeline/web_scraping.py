# -*- coding: utf-8 -*-
"""
web_scraping.py
===============
Data pipeline for Orienta.ai — Vocational Guidance Agent
Tecnológico de Monterrey · Advanced AI for Data Science II

Purpose
-------
Collects two types of external data to enrich the career recommendation engine:

1. Employer Reputation scores from QS World University Rankings
   - For UNAM and Tec de Monterrey across 17 career fields
   - Source: topuniversities.com subject rankings

2. Labor market statistics per career from IMCO's "Compara Carreras" portal
   - Employment/unemployment/informality rates
   - Average salaries broken down by gender, age, and formality
   - Cost of public/private education
   - Postgraduate premium

The final merged dataset feeds into the Orienta.ai PostgreSQL database.

Dependencies
------------
    pip install selenium pandas requests regex

Usage
-----
    python web_scraping.py

Output
------
    data_avance.csv — merged DataFrame with employer reputation + labor stats
"""

import sqlite3
import numpy as np
import pandas as pd
import json
import time
import re

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ---------------------------------------------------------------------------
# Career catalogs
# ---------------------------------------------------------------------------

UNAM_CAREERS = [
    # Area 1 — Engineering & Technology
    "Ingeniería Civil",
    "Ingeniería en Computación",
    "Ingeniería Industrial",
    "Ingeniería Eléctrica",
    "Ingeniería Mecánica",
    # Area 2 — Health Sciences
    "Médico Cirujano",
    "Psicología",
    "Biología",
    "Ingeniería Química",
    # Area 3 — Social Sciences & Business
    "Derecho",
    "Administración",
    "Economía",
    "Comunicación",
    # Area 4 — Arts & Architecture
    "Arquitectura",
    "Diseño Gráfico",
    "Urbanismo",
    "Historia",
]

TEC_CAREERS = [
    # Area 1 — Engineering & Technology
    "Ingeniería Civil",
    "Ingeniería en Tecnologías Computacionales",
    "Ingeniería Industrial y de Sistemas",
    "Ingeniería en Electrónica y Semiconductores",
    "Ingeniería Mecánica",
    # Area 2 — Health Sciences
    "Médico Cirujano",
    "Licenciatura en Psicología Clínica y de la Salud",
    "Licenciatura en Biociencias",
    "Ingeniería Química",
    # Area 3 — Social Sciences & Business
    "Licenciatura en Derecho",
    "Licenciatura en Estrategia y Transformación de Negocios",
    "Licenciatura en Finanzas",
    "Licenciatura en Comunicación y Producción Digital",
    # Area 4 — Arts & Architecture
    "Licenciatura en Arquitectura",
    "Licenciatura en Diseño",
    "Licenciatura en Urbanismo",
    "Licenciatura en Humanidades Digitales e Inteligencia Artificial",
]

# Mapping from career area → QS subject ranking URL
QS_LINKS = {
    "Ingeniería Civil":      "https://www.topuniversities.com/university-subject-rankings/civil-structural-engineering",
    "Ingeniería en Computación": "https://www.topuniversities.com/university-subject-rankings/computer-science-information-systems",
    "Ingeniería Industrial": "https://www.topuniversities.com/university-subject-rankings/mechanical-aeronautical-manufacturing-engineering",
    "Ingeniería Eléctrica":  "https://www.topuniversities.com/university-subject-rankings/electrical-electronic-engineering",
    "Ingeniería Mecánica":   "https://www.topuniversities.com/university-subject-rankings/mechanical-aeronautical-manufacturing-engineering",
    "Médico Cirujano":       "https://www.topuniversities.com/university-subject-rankings/medicine",
    "Psicología":            "https://www.topuniversities.com/university-subject-rankings/psychology",
    "Biología":              "https://www.topuniversities.com/university-subject-rankings/biological-sciences",
    "Ingeniería Química":    "https://www.topuniversities.com/university-subject-rankings/chemical-engineering",
    "Derecho":               "https://www.topuniversities.com/university-subject-rankings/law-legal-studies",
    "Administración":        "https://www.topuniversities.com/university-subject-rankings/business-management-studies",
    "Economía / Finanzas":   "https://www.topuniversities.com/university-subject-rankings/economics-econometrics",
    "Comunicación":          "https://www.topuniversities.com/university-subject-rankings/communication-media-studies",
    "Arquitectura":          "https://www.topuniversities.com/university-subject-rankings/architecture-built-environment",
    "Diseño Gráfico":        "https://www.topuniversities.com/university-subject-rankings/art-design",
    "Urbanismo":             "https://www.topuniversities.com/university-subject-rankings/architecture-built-environment",
    "Historia":              "https://www.topuniversities.com/university-subject-rankings/art-history",
}

# IMCO career slugs (comparacarreras.imco.org.mx)
IMCO_CAREER_SLUGS = [
    "construcción_e_ingeniería_civil",
    "ciencias_computacionales",
    "ingeniería_industrial",
    "electrónica_automatización_y_aplicaciones_de_la_mecánica-eléctrica",
    "mecánica_y_profesiones_afines_al_trabajo_metálico",
    "medicina_general",
    "psicología",
    "biología",
    "ingeniería_de_procesos_químicos",
    "derecho",
    "administración_de_empresas",
    "economía",
    "comunicación_y_periodismo",
    "arquitectura_y_urbanismo",
    "diseño_y_comunicación_gráfica_y_editorial",
    "arquitectura_y_urbanismo",
    "historia_y_arqueología",
]


# ---------------------------------------------------------------------------
# Helper: headless Chrome driver
# ---------------------------------------------------------------------------

def _get_driver() -> webdriver.Chrome:
    """Returns a configured headless Chrome WebDriver instance."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


# ---------------------------------------------------------------------------
# Scraper 1: QS Employer Reputation
# ---------------------------------------------------------------------------

def scrape_employer_reputation(
    university_keyword: str,
    university_id: int,
    careers: list,
    qs_links: dict,
) -> list[dict]:
    """
    Scrapes Employer Reputation scores from QS World University Rankings
    for a given university across multiple career fields.

    Parameters
    ----------
    university_keyword : str
        Search keyword used in the QS URL filter (e.g., "unam", "monterrey").
    university_id : int
        Numeric ID to identify the university in the output dataset.
    careers : list
        List of career names to iterate over.
    qs_links : dict
        Mapping from career name to QS subject ranking URL.

    Returns
    -------
    list[dict]
        One record per career with fields: id, name, university, emp_rate.
    """
    records = []
    driver = _get_driver()

    try:
        for i, career in enumerate(careers):
            base_url = list(qs_links.values())[i]
            url = f"{base_url}?search={university_keyword}"
            print(f"  Fetching: {url}")

            driver.get(url)
            time.sleep(5)  # Wait for JavaScript rendering

            page_source = driver.page_source

            if university_keyword in page_source.lower():
                try:
                    element = driver.find_element(By.CLASS_NAME, "new-rankings-ind-val")
                    emp_rate = element.text
                    print(f"    ✓ Employer Reputation: {emp_rate}")
                except Exception:
                    emp_rate = None
                    print("    ✗ Score element not found in page.")
            else:
                emp_rate = None
                print("    ✗ University not found in ranking page.")

            records.append({
                "id": i,
                "name": career,
                "university": university_id,
                "spe_fee": None,
                "emp_rate": emp_rate,
            })

    finally:
        driver.quit()
        print("  Driver closed.")

    return records


# ---------------------------------------------------------------------------
# Scraper 2: IMCO Labor Statistics
# ---------------------------------------------------------------------------

def scrape_imco_statistics(career_slugs: list) -> list[dict]:
    """
    Scrapes detailed labor market statistics from IMCO's Compara Carreras portal
    (comparacarreras.imco.org.mx) for each career slug.

    Extracted fields include:
    - Total enrolled students and new graduates
    - Public/private education cost and quality rating
    - Employment, unemployment, and informality rates
    - Quality employment probability
    - Average salary (total, by gender, by age, by formality)
    - Career ranking
    - Postgraduate percentage, salary, and salary increase

    Parameters
    ----------
    career_slugs : list
        IMCO URL slugs for each career (e.g., "ciencias_computacionales").

    Returns
    -------
    list[dict]
        One record per career with all scraped labor statistics.
    """
    records = []
    driver = _get_driver()

    try:
        for i, slug in enumerate(career_slugs):
            url = f"https://comparacarreras.imco.org.mx/{slug}"
            print(f"  Fetching: {url}")

            driver.get(url)
            time.sleep(5)

            def get_text(element_id: str) -> str:
                """Safely extract text from a DOM element by ID."""
                try:
                    return driver.find_element(By.ID, element_id).text
                except Exception:
                    return None

            def get_text_xpath(xpath: str) -> str:
                """Safely extract text from a DOM element by XPath."""
                try:
                    return driver.find_element(By.XPATH, xpath).text
                except Exception:
                    return None

            record = {
                "id": i,
                "career": slug,
                # Enrollment
                "total_students":              get_text_xpath('//*[@id="total-students"]/h3'),
                "new_graduates_number":        get_text("new-graduates-number"),
                # Cost & quality
                "public_cost":                 get_text("public-cost"),
                "private_cost":                get_text("private-cost"),
                "public_quality_rating":       get_text("public-quality-rating"),
                "private_quality_rating":      get_text("private-quality-rating"),
                # Employment
                "occupation_rate":             get_text("occupation-rate"),
                "unemployment_rate":           get_text("unemployment-rate"),
                "informality_rate":            get_text("informality-rate"),
                "quality_employment_probability": get_text("quality-employment-probability"),
                # Salary
                "average_salary":              get_text("average-salary"),
                "career_rank":                 get_text("career-rank"),
                "women_salary":                get_text("women-salary"),
                "men_salary":                  get_text("men-salary"),
                "under_30_salary":             get_text("under-30-salary"),
                "over_30_salary":              get_text("over-30-salary"),
                "formal_salary":               get_text("formal-salary"),
                "informal_salary":             get_text("informal-salary"),
                # Postgraduate
                "postgrad_percentage":         get_text("postgrad-percentage"),
                "postgrad_salary":             get_text("postgrad-salary"),
                "salary_increase":             get_text("salary-increase"),
            }

            records.append(record)
            print(f"    ✓ Avg salary: {record['average_salary']} | Employment: {record['occupation_rate']}")

    finally:
        driver.quit()
        print("  Driver closed.")

    return records


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    """
    Runs the full data collection pipeline:
      1. Scrape QS Employer Reputation for UNAM and Tec de Monterrey
      2. Scrape IMCO labor statistics for all career slugs
      3. Merge both datasets on career ID
      4. Export to CSV

    Returns
    -------
    pd.DataFrame
        Merged dataset ready for database ingestion.
    """
    print("=" * 60)
    print("STEP 1 — QS Employer Reputation: UNAM")
    print("=" * 60)
    unam_records = scrape_employer_reputation("unam", 1, UNAM_CAREERS, QS_LINKS)

    print("\n" + "=" * 60)
    print("STEP 2 — QS Employer Reputation: Tec de Monterrey")
    print("=" * 60)
    tec_records = scrape_employer_reputation("monterrey", 2, TEC_CAREERS, QS_LINKS)

    # Combine employer reputation records
    df_reputation = pd.DataFrame(unam_records + tec_records)

    print("\n" + "=" * 60)
    print("STEP 3 — IMCO Labor Statistics")
    print("=" * 60)
    imco_data = scrape_imco_statistics(IMCO_CAREER_SLUGS)
    df_imco = pd.json_normalize(imco_data)

    print("\n" + "=" * 60)
    print("STEP 4 — Merging datasets")
    print("=" * 60)
    result = pd.merge(df_reputation, df_imco, on="id")

    output_path = "data_avance.csv"
    result.to_csv(output_path, index=False)
    print(f"\n✅ Pipeline complete. Output saved to: {output_path}")
    print(f"   Shape: {result.shape}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run_pipeline()
    print(df.head())
