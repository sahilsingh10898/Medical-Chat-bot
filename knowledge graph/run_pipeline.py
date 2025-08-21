def run_all_stages(input_raw, stage1_out, stage2_out, kg_out, nebula=False):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Clinical KG Pipeline")
    parser.add_argument("--input_raw", required=True, help="Raw CSV input folder")
    parser.add_argument("--stage1_out", required=True, help="Output for cleaned step 1")
    parser.add_argument("--stage2_out", required=True, help="Output for cleaned + symptoms")
    parser.add_argument("--kg_out", required=True, help="Final KG CSV output folder")
    parser.add_argument("--nebula", action="store_true", help="Ingest into NebulaGraph")

    args = parser.parse_args()

    run_all_stages(
        input_raw=args.input_raw,
        stage1_out=args.stage1_out,
        stage2_out=args.stage2_out,
        kg_out=args.kg_out,
        nebula=args.nebula
    )