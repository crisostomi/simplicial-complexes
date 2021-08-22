import torch

## Evaluation of a set of models
#
# n_models = 3
# aggregation = "sum"
# component_to_use = "both"
# num_epochs = 400
# margins = [0.3, 0.2, 0.1, 0.05]
#
# models = []
#
# for i in range(n_models):
#     # model = DeffSCNN(colors = 1).to(device)
#     model = MySCNN(
#         filter_size,
#         colors=1,
#         aggregation=aggregation,
#         component_to_use=component_to_use,
#         keep_separated=False,
#     )
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     model.
#     models.append(model)
#
# accuracies = evaluate_accuracies_margins(
#     models, margins, inputs, targets, components, only_missing_simplices=True
# )
# summarize_accuracies(accuracies, margins)
#
# # Grid search
#
# n_models = 5
# models = []
# num_epochs = 400
# aggregations = ["MLP", "sum"]
# components_to_use = ["irr", "sol", "both"]
# margins = [0.1, 0.05, 0.02, 0.01]
#
# for comp in components_to_use:
#
#     for aggregation in aggregations:
#
#         if comp != "both" and aggregation != "sum":
#             continue
#
#         print(f"Components to use: {comp}, aggregation: {aggregation}")
#
#         for i in range(n_models):
#             model = MySCNN(
#                 filter_size, colors=1, aggregation=aggregation, component_to_use=comp
#             ).to(device)
#             optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#             train(
#                 model,
#                 num_epochs,
#                 components,
#                 inputs,
#                 known_indices,
#                 optimizer,
#                 device,
#                 verbose=False,
#             )
#             models.append(model)
#
#         accuracies = evaluate_accuracies_margins(models, margins)
#         summarize_accuracies(accuracies, margins)
