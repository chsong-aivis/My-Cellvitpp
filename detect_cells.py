if command.lower() == "process_wsi":
    celldetector.process_wsi(
        wsi_path=args["wsi_path"],
        wsi_properties=wsi_properties,
        graph=args["graph"],
    )
elif command.lower() == "process_dataset":
    if args["filelist"] is not None:
        # ... (기존 코드)
        celldetector.process_wsi(
            wsi_path=row["path"],
            wsi_properties=wsi_properties,
            graph=args["graph"],
        )
    else:
        wsi_paths = sorted(Path(args["data_dir"]).rglob("*.tiff"))
        # ... (기존 코드)
        for wsi_path in wsi_paths:
            celldetector.process_wsi(
                wsi_path=wsi_path,
                wsi_properties=wsi_properties,
                graph=args["graph"],
            )
else:
    raise ValueError("Command not recognized") 