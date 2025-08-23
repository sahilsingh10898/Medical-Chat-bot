from pathlib import Path

def run_all_stages(input_raw, stage1_out, stage2_out, kg_out, nebula=False):
    # Validate input folder exists
    if not Path(input_raw).exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_raw}")
    
    # Ensure output folders exist
    Path(stage1_out).mkdir(parents=True, exist_ok=True)
    Path(stage2_out).mkdir(parents=True, exist_ok=True)
    Path(kg_out).mkdir(parents=True, exist_ok=True)

    print("\n[Stage 1] Cleaning raw CSV files...")
    run_cleaning_process(input_raw, stage1_out)

    print("\n[Stage 2] Extracting symptoms with NER...")
    run_symptom_extraction_pipeline(stage1_out, stage2_out)

    print("\n[Stage 3] Building Knowledge Graph...")
    run_kg_builder(stage2_out, kg_out)

    if nebula:
        print("\n[Stage 4] Ingesting KG into NebulaGraph...")
        ingest_to_nebula_from_csv(csv_dir=kg_out)

    print("\n✅ All stages completed.")