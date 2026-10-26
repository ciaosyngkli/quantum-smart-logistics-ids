pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/your-username/cicd-lab1.git'
            }
        }

        stage('Setup Python Environment') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    if [ -f requirements.txt ]; then
                        pip install -r requirements.txt
                    else
                        pip install qiskit scikit-learn pytest
                    fi
                '''
            }
        }

        stage('Run Tests & Validate Model') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest || python3 quids_model.py
                '''
            }
        }
    }

    post {
        success {
            echo 'Quantum model CI pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline failed during build or test execution.'
        }
    }
}
