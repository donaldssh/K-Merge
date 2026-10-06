import numpy as np


def merge_loras_cluster(model, args, incremental_step, cluster_step, closest_cluster):

    if args.lora_merge_strategy == "kmerge":
        frac = 1 / (cluster_step)
        weights = (np.array([1, 1]) * np.array([1 - frac, frac])).tolist()
    else:
        weights = [0.5, 0.5]

    adapter_name = f"cluster_{closest_cluster}"
    proxy_adapter_name = "temp"

    if args.lora_merge_strategy in ["linear", "kmerge"] and (
        not args.use_all_models_at_step
    ):
        combination_type = "linear"

    elif (
        args.lora_merge_strategy in ["linear", "kmerge"] and args.use_all_models_at_step
    ):
        combination_type = "linear"

        weights = [
            1 / len(args.lora_merge_modules[: incremental_step + 1])
            for _ in range(len(args.lora_merge_modules[: incremental_step + 1]))
        ]

        model.add_weighted_adapter(
            args.lora_merge_modules[: incremental_step + 1],
            weights,
            proxy_adapter_name,
            combination_type=combination_type,
            density=args.density,
        )
        model.delete_adapter(adapter_name)

        # Clone merged proxy back to original name
        model.add_weighted_adapter(
            [proxy_adapter_name],
            [1.0],
            adapter_name,
            combination_type=combination_type,
            density=args.density,
        )

        # Clean up temp
        model.delete_adapter(proxy_adapter_name)

        return

    elif args.dec_peft:
        frac = 1 / (cluster_step)
        weights = (np.array([1, 1]) * np.array([1 - frac, frac])).tolist()
        combination_type = args.lora_merge_strategy[18:]

    else:
        weights = [1, 1]
        combination_type = args.lora_merge_strategy  # i.e. ties, dare_linear, dare_ties

    model.add_weighted_adapter(
        [adapter_name] + ["new_lora"],
        weights,  # real merge weights like [0.7, 0.3]
        proxy_adapter_name,
        combination_type=combination_type,
        density=args.density,
    )

    # Delete the original adapter
    model.delete_adapter(adapter_name)

    # Clone merged proxy back to original name
    model.add_weighted_adapter(
        [proxy_adapter_name],
        [1.0],
        adapter_name,
        combination_type=combination_type,
        density=args.density,
    )

    # Clean up temp
    model.delete_adapter(proxy_adapter_name)
