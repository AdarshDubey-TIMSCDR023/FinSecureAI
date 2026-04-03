document.getElementById("predict-btn").addEventListener("click", predictTransaction)

async function predictTransaction(){

const data = {

Amount: parseFloat(document.getElementById("amount").value) || 0,
Time: parseFloat(document.getElementById("time").value) || 0,
V1: parseFloat(document.getElementById("v1").value) || 0,
V2: parseFloat(document.getElementById("v2").value) || 0,
V3: parseFloat(document.getElementById("v3").value) || 0,
V4: parseFloat(document.getElementById("v4").value) || 0

}

const response = await fetch("/predict",{

method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify(data)

})

const result = await response.json()

if(!result.success){
alert(result.error)
return
}

const pred = result.prediction

const box = document.getElementById("result-box")
box.classList.remove("hidden")

document.getElementById("result-label").innerText =
pred.label==="Fraud" ? "⚠ FRAUD TRANSACTION" : "✓ NORMAL TRANSACTION"

document.getElementById("confidence").innerText =
"Confidence: "+pred.confidence+"%"

document.getElementById("risk").innerText =
"Risk Level: "+pred.risk_level

document.getElementById("probability").innerText =
"Fraud Probability: "+pred.probability

box.className="result "+(pred.label==="Fraud"?"fraud":"normal")

}