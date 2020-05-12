import numpy as np

def reptile(model, datasets, total_epochs = 750, batch_sz = 16, inner_steps = 20, epsilon = 1):
  for epoch in range(total_epochs):
      print("META EPOCH {0}".format(epoch + 1))
      old_weights = model.get_weights()
      task = np.random.randint(num_tasks)
      dataset = datasets[task]
      rnd_ord = np.arange(len(dataset["x_train"]))
      np.random.shuffle(rnd_ord)
      model.fit(dataset["x_train"][rnd_ord][:inner_steps * batch_sz],
                  dataset["y_train"][rnd_ord][:inner_steps * batch_sz],
                  batch_size = batch_sz,
                  epochs = 1, 
                  verbose = 2,
                  validation_data=(dataset["x_test"], dataset["y_test"]))
      task_weights = model.get_weights()
      meta_grad = [old - task for (old, task) in zip(old_weights, task_weights)]
      updated_weights = [old - epsilon * grad for (old, grad) in zip(old_weights, meta_grad)]
      model.set_weights(updated_weights)
