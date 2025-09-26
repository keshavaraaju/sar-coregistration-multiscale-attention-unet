# Training Pipeline (Pseudocode)

Function train_model(train_batches, val_batches, model, epochs, optimizer, loss_fn):
    for epoch in range(epochs):
        train_loss = 0
        for batch in train_batches:
            pri, sec, rg, az = unpack(batch)
            pred_rg, pred_az = model(pri, sec)
            loss_rg = loss_fn(pred_rg, rg)
            loss_az = loss_fn(pred_az, az)
            total_loss = loss_rg + loss_az

            backpropagate(total_loss)
            optimizer.update(model)
            train_loss += total_loss

        average_train_loss = train_loss / number_of_batches

        val_loss = 0
        for batch in val_batches:
            pri, sec, rg, az = unpack(batch)
            pred = model(pri, sec)
            val_loss += loss_fn(pred, (rg, az))

        average_val_loss = val_loss / number_of_val_batches

        if early_stopping(average_val_loss):
            break

    return trained_model
