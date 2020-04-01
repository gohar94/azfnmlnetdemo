using System.IO;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Azure.WebJobs.Host;
using Newtonsoft.Json;
using Microsoft.ML;
using serverlessfunctionapp.Predict;
using Microsoft.Extensions.Logging;

namespace Predict
{
    public static class Predict
    {
        [FunctionName("Predict")]
        public static IActionResult Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", Route = null)]HttpRequest req,
            [Blob("models/model.zip", FileAccess.Read, Connection = "AzureWebJobsStorage")] Stream serializedModel,
            ILogger log)
        {

             var watch = System.Diagnostics.Stopwatch.StartNew();

            // Workaround for Azure Functions Host
            if (typeof(Microsoft.ML.Runtime.Data.LoadTransform) == null ||
                typeof(Microsoft.ML.Runtime.Learners.LinearClassificationTrainer) == null ||
                typeof(Microsoft.ML.Runtime.Internal.CpuMath.SseUtils) == null ||
                typeof(Microsoft.ML.Runtime.FastTree.FastTree) == null)
            {
                log.LogError("Error loading ML.NET");
                return new StatusCodeResult(500);
            }

            //Read incoming request body
            string requestBody = new StreamReader(req.Body).ReadToEnd();

            log.LogInformation(requestBody);

            //Bind request body to IrisData object
            IrisData data = JsonConvert.DeserializeObject<IrisData>(requestBody);

            //Load prediction model
            var model = PredictionModel.ReadAsync<IrisData, IrisPrediction>(serializedModel).Result;
            serializedModel.Close();

            //Make prediction
            IrisPrediction prediction = model.Predict(data);

            watch.Stop();
            string output = "Time taken (ms): " + watch.ElapsedMilliseconds.ToString() + ", Predicted Label: " + prediction.PredictedLabels;

            //Return prediction
            return (IActionResult)new OkObjectResult(output);
        }
    }
}
