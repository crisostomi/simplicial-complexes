def summarize_accuracies(accuracies, margins):
    """
prints the mean and standard deviation of the accuracies of a set of models

parameters:
    accuracies:
    margin:
"""
    for margin in margins:
        accuracies[margin] = np.array(accuracies[margin])
        mean_accuracies = np.mean(accuracies[margin], axis=0)
        std_accuracies = np.std(accuracies[margin], axis=0)
        print(f"\t margin: {margin} mean: {mean_accuracies}, std: {std_accuracies}")


def evaluate(model, margin, components, inputs, targets, only_missing_simplices):
    """
Runs the model over the inputs and then evaluates its accuracy
"""
    known_indices_set = [set(known_indices[i]) for i in range(len(known_indices))]

    with torch.no_grad():
        inputs = [input.clone() for input in inputs]
        optimizer.zero_grad()
        preds = model(components, inputs)
        accuracies = get_accuracy(
            targets, preds, margin, known_indices_set, only_missing_simplices
        )
    return accuracies


def get_accuracy(targets, preds, margin, known_indices, only_missing_simplices):
    """
returns the accuracy for each dimension by counting the number
of hits over the total number of simplices of that dimension
if only_missing_simplices is True, then the accuracy is computed only over the missing simplices
"""
    dims = len(targets)
    accuracies = []

    for k in range(dims):

        hits = 0
        den = 0

        k_dim_simplices_true = targets[k]
        (_, num_simplices_dim_k) = preds[k].shape

        for j in range(num_simplices_dim_k):

            # if we only compute the accuracy over the missing simplices,
            # then we skip this simplex if it is known
            if only_missing_simplices:
                if j in known_indices[k]:
                    continue

            curr_value_pred = preds[k][0][j]
            curr_value_true = targets[k][0][j]
            den += 1

            if similar(curr_value_pred, curr_value_true, margin):
                hits += 1

        accuracy = round(hits / den, 4)
        accuracies.append(accuracy)

    return accuracies


def similar(val_pred, val_true, margin):
    """
Returns whether the predicted value is not further than margin*original_value from the original value
"""
    return torch.abs(val_pred - val_true) <= (margin * val_true)



def evaluate_accuracies_margins(
    models: list, margins: list, inputs, targets, components, only_missing_simplices
):
    """
Computes the accuracy of each model in each dimension for each margin

parameters:
    models: list of trained models
    margins: list of floats

return:
    accuracies: list of floats, each element is a tuple of length max_simplex_dim containing
                the model accuracy for each simplex dimension
"""
    accuracies = {margin: [] for margin in margins}

    for model in models:
        for margin in margins:
            model_accuracies = evaluate(
                model, margin, components, inputs, targets, only_missing_simplices
            )
            accuracies[margin].append(model_accuracies)

    return accuracies