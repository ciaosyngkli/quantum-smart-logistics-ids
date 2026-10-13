pipeline {
    agent any

    environment {
        PYTHON_ENV = "${WORKSPACE}/.venv"
    }

    stages {
        stage('Checkout Source') {
            steps {
                checkout scm
            }
        }

        stage('Build & Dependencies') {
            steps {
                sh '''
                    python3 -m venv ${PYTHON_ENV}
                    . ${PYTHON_ENV}/bin/activate
                    pip install --upgrade pip setuptools
                    pip install numpy pandas scikit-learn qiskit qiskit-machine-learning matplotlib pytest pyinstaller
                    
                    # Bytecode compilation verification
                    python3 -m py_compile qids_hybrid_v3.py
                    
                    # Package into single executable artifact
                    pyinstaller --onefile --name quids_model --collect-all pandas --collect-all numpy --collect-all qiskit qids_hybrid_v3.py
                '''
            }
        }

        stage('Execute Unit & Integration Tests') {
            steps {
                sh '''
                    . ${PYTHON_ENV}/bin/activate
                    mkdir -p reports
                    pytest --junitxml=reports/test-results.xml || true
                '''
            }
        }
    }

    post {
        always {
            // Lab 7: Parse XML test reports in Jenkins UI
            junit 'reports/*.xml'
            
            // Lab 6: Archive the final build artifact (the binary executable)
            archiveArtifacts artifacts: 'dist/quids_model', fingerprint: true
        }
    }
}
