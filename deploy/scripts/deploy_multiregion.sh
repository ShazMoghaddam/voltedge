#!/usr/bin/env bash
# VoltEdge — Multi-Region Deployment Script
# Usage: ./deploy_multiregion.sh <image_tag> <environment>
set -euo pipefail

IMAGE_TAG="${1:-latest}"
ENVIRONMENT="${2:-production}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.eu-west-1.amazonaws.com"

REGIONS=(
  "eu-west-1:voltedge-eks-eu:voltedge-data-eu-${ENVIRONMENT}"
  "us-east-1:voltedge-eks-us:voltedge-data-us-${ENVIRONMENT}"
  "ap-southeast-1:voltedge-eks-apac:voltedge-data-apac-${ENVIRONMENT}"
)

deploy_region() {
  local region="$1" cluster="$2" s3_bucket="$3"
  echo "── Deploying to ${region} / ${cluster} ──"

  aws eks update-kubeconfig --region "$region" --name "$cluster"

  kubectl create configmap voltedge-config \
    --from-literal=environment="$ENVIRONMENT" \
    --from-literal=aws_region="$region" \
    --from-literal=s3_bucket="$s3_bucket" \
    --from-literal=log_level="INFO" \
    --from-literal=log_format="json" \
    --namespace voltedge --dry-run=client -o yaml | kubectl apply -f -

  sed "s|\${ECR_REGISTRY}|${ECR_REGISTRY}|g; s|\${IMAGE_TAG}|${IMAGE_TAG}|g" \
    deploy/k8s/deployment.yaml | kubectl apply -f -

  kubectl rollout status deployment/voltedge-api -n voltedge --timeout=300s

  local lb
  lb=$(kubectl get svc voltedge-api -n voltedge \
    -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)

  if [[ -n "$lb" ]]; then
    for i in $(seq 1 12); do
      curl -sf "http://${lb}/health" | grep -q '"status":"ok"' && \
        { echo "✅ ${region} healthy"; return 0; }
      sleep 10
    done
    echo "❌ ${region} health check failed"; return 1
  fi
}

echo "🚀 VoltEdge multi-region deploy — ${IMAGE_TAG} → ${ENVIRONMENT}"

IFS=':' read -r region cluster bucket <<< "${REGIONS[0]}"
deploy_region "$region" "$cluster" "$bucket"

for spec in "${REGIONS[@]:1}"; do
  IFS=':' read -r region cluster bucket <<< "$spec"
  echo "Pausing 30s (monitoring window)..."
  sleep 30
  deploy_region "$region" "$cluster" "$bucket"
done

echo "✅ All regions deployed: ${IMAGE_TAG} → ${ENVIRONMENT}"
