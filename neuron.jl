using Flux
using Plots

# Target function
target_function(x) = 4x + 2

# Create Artifical Data
x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = target_function.(x_train), target_function.(x_test)

# Visualise the Data
Plots.scatter(x_train, y_train, color="magenta", legend=false)

# Define simple model with one input node
model = Flux.Dense(1 => 1)

println("model weight variable = $(model.weight)")
println("model bias variable = $(model.bias)")

# Predict using training data
predictions = model(x_train)
println("predictions = $(predictions)")

Plots.scatter(x_train, predictions, color="magenta", legend=false)

# Define our loss function for optimising
loss(x, y) = Flux.Losses.mse(model(x), y)

println("Current loss = $(loss(x_train, y_train))")

# Train and update the single neurons weights and biase
# to minimise the loss function we need to define an optimiser,
# such as gradient descent

# Define gradient descent optimiser
opt = Flux.Descent()

# Format your data
data = [(x_train, y_train)]

# Collect weights and bias for your models
parameters = Flux.params(model)

# Train the model
Flux.train!(loss, parameters, data, opt)

println("New loss = $(loss(x_train, y_train))")

println("Old Loss = $(loss(x_train, y_train))")

N = length(data)

for epoch in 1:1_1000
  i = rand(1:N)
  Flux.train!(loss, parameters, [data[i]], opt)
end

println("New loss = $(loss(x_train, y_train))")

# Result

scatter(x_test, y_test, color="magenta")
domain = LinRange(6, 10, 100)

plot!(domain, domain .* model.weight .+ model.bias, legend=false)
