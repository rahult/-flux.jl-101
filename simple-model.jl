using Flux, Plots

actual(x) = 4x + 2

x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = actual.(x_train), actual.(x_test)

domain = x_train
range = y_train

scatter(domain, range, color=:green, legend=false,
  title="Actual", xlabel="domain", ylabel="range")

# model training
predict = model = Dense(1 => 1)

model.weight
model.bias

# activation(Wx+b)

predicted_y = predict(x_train)

scatter!(domain, predicted_y, color=:red, legend=false)

loss(x, y) = Flux.Losses.mse(predict(x), y)
optimiser = Flux.Descent()
data = [(x_train, y_train)]
parameters = Flux.params(model)

Flux.train!(loss, parameters, data, optimiser)

predicted_y = predict(x_train)

scatter!(domain, predicted_y, color=:red, legend=false)

predicted_values = []

for epoch in 1:1_000
  Flux.train!(loss, parameters, data, optimiser)
  local predicted_y = predict(x_train)
  push!(predicted_values, predicted_y)
end

anim = @animate for i ∈ 1:1_000
  scatter!(domain, predicted_values[i], color=:red, legend=false)
end

gif(anim, "simple_mode_training.gif", fps=30)

predicted_y = predict(x_train)

scatter!(domain, predicted_y, color=:red, legend=false)